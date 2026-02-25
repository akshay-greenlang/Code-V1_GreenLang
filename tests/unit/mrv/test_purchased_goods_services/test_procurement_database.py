"""
Test suite for ProcurementDatabaseEngine - AGENT-MRV-014

This module tests the ProcurementDatabaseEngine for the Purchased Goods & Services Agent.
Tests cover EEIO factor lookups, physical EF lookups, classification mapping,
currency conversion, margin adjustments, and supplier EF registry.

Coverage target: 85%+
Test count: 60+ tests
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional

from greenlang.purchased_goods_services.procurement_database import (
    ProcurementDatabaseEngine,
    EEIOFactor,
    PhysicalEmissionFactor,
    ClassificationMapping,
    CurrencyConversion,
    MarginAdjustment,
    SupplierEmissionFactor,
    EFHierarchy
)


class TestProcurementDatabaseSingleton:
    """Test singleton pattern for ProcurementDatabaseEngine."""

    def test_singleton_creation(self):
        """Test that engine can be created."""
        engine = ProcurementDatabaseEngine()
        assert engine is not None
        assert isinstance(engine, ProcurementDatabaseEngine)

    def test_singleton_identity(self):
        """Test that multiple calls return same instance."""
        engine1 = ProcurementDatabaseEngine()
        engine2 = ProcurementDatabaseEngine()
        assert engine1 is engine2

    def test_singleton_reset(self):
        """Test that singleton can be reset."""
        engine1 = ProcurementDatabaseEngine()
        ProcurementDatabaseEngine._instance = None
        engine2 = ProcurementDatabaseEngine()
        assert engine1 is not engine2


class TestEEIOFactorLookup:
    """Test EEIO emission factor lookups."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ProcurementDatabaseEngine()

    def test_exact_match_6digit_naics(self, engine):
        """Test exact match for 6-digit NAICS code."""
        factor = engine.lookup_eeio_factor(
            naics_code="334111",
            database="USEEIO",
            base_year=2017
        )
        assert factor is not None
        assert factor.naics_code == "334111"
        assert factor.database == "USEEIO"
        assert factor.emission_factor > Decimal("0")
        assert factor.unit == "kgCO2e/USD"

    def test_fallback_to_4digit_naics(self, engine):
        """Test fallback from 6-digit to 4-digit NAICS."""
        # Try specific 6-digit that doesn't exist
        factor = engine.lookup_eeio_factor(
            naics_code="334119",
            database="USEEIO",
            base_year=2017
        )
        assert factor is not None
        # Should fallback to 3341 (4-digit)
        assert factor.naics_code.startswith("3341")
        assert len(factor.naics_code) <= 4

    def test_fallback_to_2digit_naics(self, engine):
        """Test fallback from 6-digit to 2-digit NAICS."""
        # Try NAICS with no 4-digit match
        factor = engine.lookup_eeio_factor(
            naics_code="999999",
            database="USEEIO",
            base_year=2017
        )
        assert factor is not None
        # Should fallback to 99 (2-digit)
        assert len(factor.naics_code) == 2

    def test_missing_naics_code(self, engine):
        """Test handling of completely missing NAICS code."""
        factor = engine.lookup_eeio_factor(
            naics_code="000000",
            database="USEEIO",
            base_year=2017
        )
        # Should return None or fallback to economy-wide average
        if factor:
            assert factor.naics_code == "00"  # Economy-wide fallback
            assert factor.description == "Economy-wide average"

    def test_useeio_database(self, engine):
        """Test USEEIO database lookup."""
        factor = engine.lookup_eeio_factor(
            naics_code="311111",  # Dog and cat food manufacturing
            database="USEEIO",
            base_year=2017
        )
        assert factor is not None
        assert factor.database == "USEEIO"
        assert factor.base_year == 2017

    def test_exiobase_database(self, engine):
        """Test EXIOBASE database lookup."""
        factor = engine.lookup_eeio_factor(
            naics_code="311111",
            database="EXIOBASE",
            base_year=2015
        )
        assert factor is not None
        assert factor.database == "EXIOBASE"
        assert factor.base_year == 2015

    def test_oecd_database(self, engine):
        """Test OECD database lookup."""
        factor = engine.lookup_eeio_factor(
            naics_code="311111",
            database="OECD",
            base_year=2018
        )
        assert factor is not None
        assert factor.database == "OECD"

    def test_defra_database(self, engine):
        """Test DEFRA database lookup."""
        factor = engine.lookup_eeio_factor(
            naics_code="311111",
            database="DEFRA",
            base_year=2021
        )
        assert factor is not None
        assert factor.database == "DEFRA"

    def test_search_eeio_factors(self, engine):
        """Test searching EEIO factors by keyword."""
        factors = engine.search_eeio_factors(
            query="food manufacturing",
            database="USEEIO"
        )
        assert len(factors) > 0
        assert all(factor.database == "USEEIO" for factor in factors)
        assert all("food" in factor.description.lower() or
                  "manufacturing" in factor.description.lower()
                  for factor in factors)

    def test_get_all_eeio_factors(self, engine):
        """Test getting all EEIO factors for a database."""
        factors = engine.get_all_eeio_factors(database="USEEIO")
        assert len(factors) > 100  # USEEIO has hundreds of sectors
        assert all(factor.database == "USEEIO" for factor in factors)


class TestPhysicalEFLookup:
    """Test physical emission factor lookups."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ProcurementDatabaseEngine()

    def test_exact_match_material(self, engine):
        """Test exact match for material key."""
        factor = engine.lookup_physical_ef(
            material_key="steel_hot_rolled",
            region="GLOBAL"
        )
        assert factor is not None
        assert factor.material_key == "steel_hot_rolled"
        assert factor.emission_factor > Decimal("0")
        assert factor.unit == "kgCO2e/kg"

    def test_material_by_category(self, engine):
        """Test lookup by material category."""
        factors = engine.get_physical_efs_by_category(
            category="metals"
        )
        assert len(factors) > 0
        assert all(factor.category == "metals" for factor in factors)

    def test_regional_variation(self, engine):
        """Test regional variation in emission factors."""
        global_factor = engine.lookup_physical_ef(
            material_key="steel_hot_rolled",
            region="GLOBAL"
        )
        china_factor = engine.lookup_physical_ef(
            material_key="steel_hot_rolled",
            region="CHINA"
        )
        assert global_factor is not None
        assert china_factor is not None
        # China steel typically has higher EF than global average
        assert china_factor.emission_factor >= global_factor.emission_factor

    def test_all_materials(self, engine):
        """Test getting all physical emission factors."""
        factors = engine.get_all_physical_efs()
        assert len(factors) > 50  # Should have many materials
        # Check diversity of categories
        categories = set(f.category for f in factors)
        assert "metals" in categories
        assert "plastics" in categories
        assert "construction" in categories

    def test_missing_material_key(self, engine):
        """Test handling of missing material key."""
        factor = engine.lookup_physical_ef(
            material_key="nonexistent_material",
            region="GLOBAL"
        )
        assert factor is None

    def test_missing_region_fallback(self, engine):
        """Test fallback to GLOBAL when region not found."""
        factor = engine.lookup_physical_ef(
            material_key="steel_hot_rolled",
            region="MARS"  # Non-existent region
        )
        # Should fallback to GLOBAL
        assert factor is not None
        assert factor.region == "GLOBAL"

    def test_material_description(self, engine):
        """Test that material descriptions are present."""
        factor = engine.lookup_physical_ef(
            material_key="steel_hot_rolled",
            region="GLOBAL"
        )
        assert factor is not None
        assert factor.description
        assert len(factor.description) > 10

    def test_data_quality_indicators(self, engine):
        """Test that DQI fields are populated."""
        factor = engine.lookup_physical_ef(
            material_key="steel_hot_rolled",
            region="GLOBAL"
        )
        assert factor is not None
        assert factor.data_quality is not None
        assert 1 <= factor.data_quality <= 5


class TestClassificationMapping:
    """Test classification code mapping (NAICS, ISIC, NACE, UNSPSC)."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ProcurementDatabaseEngine()

    def test_naics_to_isic(self, engine):
        """Test NAICS to ISIC mapping."""
        isic = engine.map_classification(
            source_code="334111",
            source_system="NAICS",
            target_system="ISIC"
        )
        assert isic is not None
        assert isic.startswith("26")  # Electronics manufacturing

    def test_nace_to_isic(self, engine):
        """Test NACE to ISIC mapping."""
        isic = engine.map_classification(
            source_code="26.11",
            source_system="NACE",
            target_system="ISIC"
        )
        assert isic is not None
        assert isic.startswith("26")

    def test_isic_to_naics(self, engine):
        """Test ISIC to NAICS mapping."""
        naics = engine.map_classification(
            source_code="2611",
            source_system="ISIC",
            target_system="NAICS"
        )
        assert naics is not None
        assert naics.startswith("334")

    def test_unspsc_to_naics(self, engine):
        """Test UNSPSC to NAICS mapping."""
        naics = engine.map_classification(
            source_code="43211503",  # Computer equipment
            source_system="UNSPSC",
            target_system="NAICS"
        )
        assert naics is not None
        assert naics.startswith("334")

    def test_resolve_naics_from_naics(self, engine):
        """Test resolve_naics_from_any with NAICS input."""
        result = engine.resolve_naics_from_any(
            code="334111",
            system="NAICS"
        )
        assert result == "334111"

    def test_resolve_naics_from_nace(self, engine):
        """Test resolve_naics_from_any with NACE input."""
        result = engine.resolve_naics_from_any(
            code="26.11",
            system="NACE"
        )
        assert result is not None
        assert result.startswith("334")

    def test_resolve_naics_from_isic(self, engine):
        """Test resolve_naics_from_any with ISIC input."""
        result = engine.resolve_naics_from_any(
            code="2611",
            system="ISIC"
        )
        assert result is not None
        assert result.startswith("334")

    def test_resolve_naics_from_unspsc(self, engine):
        """Test resolve_naics_from_any with UNSPSC input."""
        result = engine.resolve_naics_from_any(
            code="43211503",
            system="UNSPSC"
        )
        assert result is not None
        assert result.startswith("334")

    def test_mapping_confidence_score(self, engine):
        """Test that mapping returns confidence score."""
        mapping = engine.get_classification_mapping(
            source_code="334111",
            source_system="NAICS",
            target_system="ISIC"
        )
        assert mapping is not None
        assert hasattr(mapping, "confidence")
        assert 0.0 <= mapping.confidence <= 1.0

    def test_one_to_many_mapping(self, engine):
        """Test handling of one-to-many mappings."""
        mappings = engine.get_all_mappings(
            source_code="334111",
            source_system="NAICS",
            target_system="ISIC"
        )
        # NAICS can map to multiple ISIC codes
        assert len(mappings) >= 1


class TestCurrencyConversion:
    """Test currency conversion and PPP adjustments."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ProcurementDatabaseEngine()

    def test_usd_to_usd_identity(self, engine):
        """Test USD to USD conversion is identity."""
        result = engine.convert_currency(
            amount=Decimal("1000.00"),
            from_currency="USD",
            to_currency="USD",
            conversion_date=datetime(2023, 6, 15)
        )
        assert result == Decimal("1000.00")

    def test_eur_to_usd_conversion(self, engine):
        """Test EUR to USD conversion."""
        result = engine.convert_currency(
            amount=Decimal("1000.00"),
            from_currency="EUR",
            to_currency="USD",
            conversion_date=datetime(2023, 6, 15)
        )
        # EUR typically stronger than USD, so result > 1000
        assert result > Decimal("1000.00")
        assert result < Decimal("1500.00")  # Sanity check

    def test_jpy_to_usd_conversion(self, engine):
        """Test JPY to USD conversion."""
        result = engine.convert_currency(
            amount=Decimal("100000.00"),
            from_currency="JPY",
            to_currency="USD",
            conversion_date=datetime(2023, 6, 15)
        )
        # JPY much weaker than USD
        assert result < Decimal("1000.00")
        assert result > Decimal("500.00")

    def test_all_supported_currencies(self, engine):
        """Test conversion for all 20 supported currencies."""
        currencies = [
            "USD", "EUR", "GBP", "JPY", "CNY", "INR", "CAD", "AUD",
            "CHF", "SEK", "NOK", "DKK", "BRL", "MXN", "ZAR", "RUB",
            "KRW", "SGD", "HKD", "NZD"
        ]
        for currency in currencies:
            result = engine.convert_currency(
                amount=Decimal("1000.00"),
                from_currency=currency,
                to_currency="USD",
                conversion_date=datetime(2023, 6, 15)
            )
            assert result > Decimal("0")

    def test_deflation_to_base_year(self, engine):
        """Test deflation to EEIO base year (2017)."""
        deflated = engine.deflate_to_base_year(
            amount=Decimal("1000.00"),
            currency="USD",
            current_year=2023,
            base_year=2017
        )
        # 2023 dollars should be worth less than 2017 dollars (inflation)
        assert deflated < Decimal("1000.00")
        assert deflated > Decimal("800.00")  # Reasonable range

    def test_ppp_adjustment(self, engine):
        """Test PPP (purchasing power parity) adjustment."""
        ppp_adjusted = engine.convert_currency_ppp(
            amount=Decimal("1000.00"),
            from_country="CHINA",
            to_country="USA",
            year=2023
        )
        # PPP-adjusted values differ from market exchange rates
        assert ppp_adjusted > Decimal("0")
        # China has lower PPP ratio, so PPP-adjusted should be higher
        assert ppp_adjusted > Decimal("1000.00")

    def test_missing_exchange_rate(self, engine):
        """Test handling of missing exchange rate."""
        with pytest.raises(ValueError, match="Exchange rate not found"):
            engine.convert_currency(
                amount=Decimal("1000.00"),
                from_currency="XYZ",  # Non-existent currency
                to_currency="USD",
                conversion_date=datetime(2023, 6, 15)
            )

    def test_historical_exchange_rate(self, engine):
        """Test historical exchange rate lookup."""
        result_2020 = engine.convert_currency(
            amount=Decimal("1000.00"),
            from_currency="EUR",
            to_currency="USD",
            conversion_date=datetime(2020, 6, 15)
        )
        result_2023 = engine.convert_currency(
            amount=Decimal("1000.00"),
            from_currency="EUR",
            to_currency="USD",
            conversion_date=datetime(2023, 6, 15)
        )
        # Exchange rates change over time
        assert result_2020 != result_2023


class TestMarginAdjustment:
    """Test purchaser price to producer price margin adjustments."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ProcurementDatabaseEngine()

    def test_lookup_margin_by_sector(self, engine):
        """Test margin lookup by NAICS sector."""
        margin = engine.lookup_margin(naics_code="334111")
        assert margin is not None
        assert Decimal("0") <= margin.margin_rate <= Decimal("1")
        assert margin.naics_code.startswith("334")

    def test_remove_margin_from_spend(self, engine):
        """Test removing retail/transport margins from spend."""
        producer_price = engine.remove_margin(
            purchaser_price=Decimal("1000.00"),
            naics_code="334111"
        )
        # Producer price should be lower (margins removed)
        assert producer_price < Decimal("1000.00")
        assert producer_price > Decimal("700.00")  # Reasonable range

    def test_prepare_spend_record(self, engine):
        """Test preparing spend record with margin removal."""
        result = engine.prepare_spend_record(
            spend_amount=Decimal("1000.00"),
            naics_code="334111",
            currency="USD",
            spend_year=2023
        )
        assert result["producer_price"] < Decimal("1000.00")
        assert result["purchaser_price"] == Decimal("1000.00")
        assert result["margin_removed"] > Decimal("0")
        assert result["margin_rate"] > Decimal("0")

    def test_missing_sector_fallback(self, engine):
        """Test fallback margin for missing sector."""
        margin = engine.lookup_margin(naics_code="999999")
        # Should use economy-wide average margin
        assert margin is not None
        assert margin.naics_code == "00"  # Economy-wide
        assert Decimal("0.1") <= margin.margin_rate <= Decimal("0.3")

    def test_zero_margin_sectors(self, engine):
        """Test sectors with near-zero margins (e.g., commodities)."""
        margin = engine.lookup_margin(naics_code="211111")  # Oil extraction
        assert margin is not None
        # Extraction industries have low margins
        assert margin.margin_rate < Decimal("0.1")

    def test_high_margin_sectors(self, engine):
        """Test sectors with high margins (e.g., retail)."""
        margin = engine.lookup_margin(naics_code="452000")  # General retail
        assert margin is not None
        # Retail has higher margins
        assert margin.margin_rate > Decimal("0.2")


class TestSupplierEFRegistry:
    """Test supplier-specific emission factor registry."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ProcurementDatabaseEngine()

    def test_register_supplier_ef(self, engine):
        """Test registering supplier-specific EF."""
        engine.register_supplier_ef(
            supplier_id="SUP-001",
            product_category="electronics",
            emission_factor=Decimal("1.25"),
            unit="kgCO2e/USD",
            data_source="supplier_epd",
            valid_from=datetime(2023, 1, 1),
            valid_to=datetime(2024, 12, 31)
        )
        # Lookup should succeed
        factor = engine.lookup_supplier_ef(
            supplier_id="SUP-001",
            product_category="electronics"
        )
        assert factor is not None
        assert factor.emission_factor == Decimal("1.25")

    def test_lookup_supplier_ef(self, engine):
        """Test looking up supplier-specific EF."""
        # Register first
        engine.register_supplier_ef(
            supplier_id="SUP-002",
            product_category="steel",
            emission_factor=Decimal("2.10"),
            unit="kgCO2e/kg",
            data_source="supplier_report"
        )
        factor = engine.lookup_supplier_ef(
            supplier_id="SUP-002",
            product_category="steel"
        )
        assert factor is not None
        assert factor.supplier_id == "SUP-002"
        assert factor.product_category == "steel"

    def test_get_all_supplier_efs(self, engine):
        """Test getting all EFs for a supplier."""
        # Register multiple
        engine.register_supplier_ef(
            supplier_id="SUP-003",
            product_category="electronics",
            emission_factor=Decimal("1.30"),
            unit="kgCO2e/USD",
            data_source="supplier_epd"
        )
        engine.register_supplier_ef(
            supplier_id="SUP-003",
            product_category="plastics",
            emission_factor=Decimal("3.50"),
            unit="kgCO2e/kg",
            data_source="supplier_epd"
        )
        factors = engine.get_supplier_efs(supplier_id="SUP-003")
        assert len(factors) == 2
        categories = [f.product_category for f in factors]
        assert "electronics" in categories
        assert "plastics" in categories

    def test_missing_supplier_ef(self, engine):
        """Test handling of missing supplier EF."""
        factor = engine.lookup_supplier_ef(
            supplier_id="SUP-999",
            product_category="nonexistent"
        )
        assert factor is None

    def test_expired_supplier_ef(self, engine):
        """Test handling of expired supplier EF."""
        engine.register_supplier_ef(
            supplier_id="SUP-004",
            product_category="aluminum",
            emission_factor=Decimal("8.50"),
            unit="kgCO2e/kg",
            data_source="supplier_report",
            valid_from=datetime(2020, 1, 1),
            valid_to=datetime(2022, 12, 31)
        )
        # Lookup with current date should return None (expired)
        factor = engine.lookup_supplier_ef(
            supplier_id="SUP-004",
            product_category="aluminum",
            as_of_date=datetime(2023, 6, 15)
        )
        assert factor is None

    def test_supplier_ef_priority(self, engine):
        """Test that supplier EF has priority over generic EF."""
        # Register supplier-specific
        engine.register_supplier_ef(
            supplier_id="SUP-005",
            product_category="steel",
            emission_factor=Decimal("1.80"),
            unit="kgCO2e/kg",
            data_source="supplier_epd"
        )
        # Should prefer supplier-specific over generic
        factor = engine.get_best_ef(
            supplier_id="SUP-005",
            material_key="steel_hot_rolled",
            naics_code="331110"
        )
        assert factor.source == "supplier_epd"
        assert factor.emission_factor == Decimal("1.80")

    def test_update_supplier_ef(self, engine):
        """Test updating existing supplier EF."""
        engine.register_supplier_ef(
            supplier_id="SUP-006",
            product_category="cement",
            emission_factor=Decimal("0.90"),
            unit="kgCO2e/kg",
            data_source="supplier_report"
        )
        # Update
        engine.register_supplier_ef(
            supplier_id="SUP-006",
            product_category="cement",
            emission_factor=Decimal("0.85"),
            unit="kgCO2e/kg",
            data_source="supplier_epd"
        )
        factor = engine.lookup_supplier_ef(
            supplier_id="SUP-006",
            product_category="cement"
        )
        assert factor.emission_factor == Decimal("0.85")
        assert factor.data_source == "supplier_epd"


class TestEFHierarchy:
    """Test emission factor selection hierarchy."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ProcurementDatabaseEngine()

    def test_select_best_ef_supplier_first(self, engine):
        """Test that supplier-specific EF is selected first."""
        # Register supplier EF
        engine.register_supplier_ef(
            supplier_id="SUP-100",
            product_category="steel",
            emission_factor=Decimal("1.50"),
            unit="kgCO2e/kg",
            data_source="supplier_epd"
        )
        best = engine.select_best_ef(
            supplier_id="SUP-100",
            material_key="steel_hot_rolled",
            naics_code="331110",
            region="USA"
        )
        assert best.hierarchy_level == 1  # Supplier-specific
        assert best.emission_factor == Decimal("1.50")

    def test_select_best_ef_product_second(self, engine):
        """Test product-specific EF when no supplier EF."""
        best = engine.select_best_ef(
            supplier_id="SUP-999",  # No supplier EF
            material_key="steel_hot_rolled",
            naics_code="331110",
            region="USA"
        )
        assert best.hierarchy_level == 2  # Product-specific
        assert best.material_key == "steel_hot_rolled"

    def test_select_best_ef_hybrid_third(self, engine):
        """Test hybrid (spend + physical) when no product EF."""
        best = engine.select_best_ef(
            supplier_id="SUP-999",
            material_key=None,  # No material key
            naics_code="331110",
            region="USA"
        )
        assert best.hierarchy_level == 3  # Hybrid
        assert best.naics_code == "331110"

    def test_select_best_ef_eeio_fourth(self, engine):
        """Test EEIO fallback when no physical data."""
        best = engine.select_best_ef(
            supplier_id="SUP-999",
            material_key=None,
            naics_code="541511",  # Services (no physical product)
            region="USA"
        )
        assert best.hierarchy_level == 4  # EEIO only
        assert best.naics_code.startswith("5415")

    def test_select_best_ef_economy_wide_last(self, engine):
        """Test economy-wide average as last resort."""
        best = engine.select_best_ef(
            supplier_id="SUP-999",
            material_key=None,
            naics_code="999999",  # Invalid code
            region="USA"
        )
        assert best.hierarchy_level == 5  # Economy-wide average
        assert best.naics_code == "00"


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ProcurementDatabaseEngine()

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
            "eeio_factor_count",
            "physical_ef_count",
            "supplier_ef_count",
            "currency_pair_count",
            "classification_mapping_count",
            "margin_adjustment_count",
            "last_updated"
        ]
        for field in expected_fields:
            assert field in health
            assert health[field] is not None
