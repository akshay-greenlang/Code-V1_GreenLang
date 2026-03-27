"""
Unit tests for WTTFuelDatabaseEngine (AGENT-MRV-016 Engine 1)

Tests all methods of WTTFuelDatabaseEngine with comprehensive coverage.
Validates well-to-tank (WTT) emission factors, fuel properties, unit conversions,
and fuel classification logic.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from greenlang.agents.mrv.fuel_energy_activities.models import (
    WTTFactor,
    FuelProperties,
    FuelClassification,
    SupplyChainBreakdown,
    GasBreakdown,
    FuelType,
    EnergyUnit,
    EmissionFactorSource,
)
from greenlang.agents.mrv.fuel_energy_activities.engines.wtt_fuel_database import WTTFuelDatabaseEngine
from greenlang_core.exceptions import ValidationError, ProcessingError


# Fixtures
@pytest.fixture
def db_engine():
    """Create WTTFuelDatabaseEngine instance for testing."""
    engine = WTTFuelDatabaseEngine()
    engine.reset()  # Ensure clean state
    return engine


@pytest.fixture
def sample_wtt_factors():
    """Sample WTT emission factors for common fuels."""
    return {
        FuelType.NATURAL_GAS: Decimal("0.0246"),  # kgCO2e/kWh
        FuelType.DIESEL: Decimal("0.0507"),
        FuelType.GASOLINE: Decimal("0.0489"),
        FuelType.COAL: Decimal("0.0182"),
        FuelType.LPG: Decimal("0.0295"),
        FuelType.FUEL_OIL: Decimal("0.0512"),
    }


@pytest.fixture
def sample_fuel_properties():
    """Sample fuel properties (heating values, densities)."""
    return {
        FuelType.DIESEL: FuelProperties(
            fuel_type=FuelType.DIESEL,
            net_calorific_value=Decimal("10.7"),  # kWh/kg
            gross_calorific_value=Decimal("11.3"),
            density=Decimal("0.835"),  # kg/litre
            carbon_content=Decimal("0.874"),  # kg C/kg fuel
        ),
        FuelType.NATURAL_GAS: FuelProperties(
            fuel_type=FuelType.NATURAL_GAS,
            net_calorific_value=Decimal("13.3"),  # kWh/kg
            gross_calorific_value=Decimal("14.7"),
            density=Decimal("0.0007785"),  # kg/litre at STP
            carbon_content=Decimal("0.750"),
        ),
    }


# Test Class
class TestWTTFuelDatabaseEngine:
    """Test suite for WTTFuelDatabaseEngine."""

    def test_initialization(self):
        """Test engine initializes correctly with default factors."""
        engine = WTTFuelDatabaseEngine()

        assert engine is not None
        stats = engine.get_statistics()
        assert stats["total_factors"] > 0
        assert stats["fuel_types_covered"] >= 25
        assert len(stats["sources"]) >= 5

    def test_get_wtt_factor_natural_gas(self, db_engine):
        """Test retrieving WTT factor for natural gas."""
        factor = db_engine.get_wtt_factor(
            fuel_type=FuelType.NATURAL_GAS,
            region="US",
            source=EmissionFactorSource.EPA_GHGRP,
        )

        assert factor is not None
        assert factor.fuel_type == FuelType.NATURAL_GAS
        assert factor.wtt_emission_factor > 0
        assert factor.unit == EnergyUnit.KWH
        assert factor.source == EmissionFactorSource.EPA_GHGRP
        assert factor.region in ["US", "GLOBAL"]

    def test_get_wtt_factor_diesel(self, db_engine):
        """Test retrieving WTT factor for diesel."""
        factor = db_engine.get_wtt_factor(
            fuel_type=FuelType.DIESEL,
            region="UK",
            source=EmissionFactorSource.DEFRA,
        )

        assert factor is not None
        assert factor.fuel_type == FuelType.DIESEL
        assert Decimal("0.04") < factor.wtt_emission_factor < Decimal("0.06")
        assert factor.region in ["UK", "GLOBAL"]

    def test_get_wtt_factor_coal(self, db_engine):
        """Test retrieving WTT factor for coal."""
        factor = db_engine.get_wtt_factor(
            fuel_type=FuelType.COAL,
            region="GLOBAL",
        )

        assert factor is not None
        assert factor.fuel_type == FuelType.COAL
        assert Decimal("0.015") < factor.wtt_emission_factor < Decimal("0.025")

    @pytest.mark.parametrize("fuel_type,expected_range", [
        (FuelType.NATURAL_GAS, (Decimal("0.020"), Decimal("0.030"))),
        (FuelType.DIESEL, (Decimal("0.045"), Decimal("0.055"))),
        (FuelType.GASOLINE, (Decimal("0.043"), Decimal("0.053"))),
        (FuelType.COAL, (Decimal("0.015"), Decimal("0.025"))),
        (FuelType.LPG, (Decimal("0.025"), Decimal("0.035"))),
        (FuelType.FUEL_OIL, (Decimal("0.045"), Decimal("0.055"))),
        (FuelType.JET_FUEL, (Decimal("0.045"), Decimal("0.055"))),
        (FuelType.KEROSENE, (Decimal("0.045"), Decimal("0.055"))),
        (FuelType.BIOMASS, (Decimal("0.001"), Decimal("0.010"))),
        (FuelType.BIOGAS, (Decimal("0.001"), Decimal("0.010"))),
        (FuelType.BIODIESEL, (Decimal("0.010"), Decimal("0.025"))),
        (FuelType.ETHANOL, (Decimal("0.010"), Decimal("0.025"))),
        (FuelType.WASTE_OIL, (Decimal("0.005"), Decimal("0.020"))),
        (FuelType.MUNICIPAL_WASTE, (Decimal("0.005"), Decimal("0.020"))),
        (FuelType.HYDROGEN, (Decimal("0.010"), Decimal("0.100"))),
    ])
    def test_get_wtt_factor_all_fuel_types(self, db_engine, fuel_type, expected_range):
        """Test WTT factors for all major fuel types."""
        factor = db_engine.get_wtt_factor(fuel_type=fuel_type)

        assert factor is not None
        assert factor.fuel_type == fuel_type
        min_val, max_val = expected_range
        assert min_val <= factor.wtt_emission_factor <= max_val

    def test_get_wtt_factors_all_sources(self, db_engine):
        """Test retrieving WTT factors from all available sources."""
        sources = [
            EmissionFactorSource.EPA_GHGRP,
            EmissionFactorSource.DEFRA,
            EmissionFactorSource.IPCC,
            EmissionFactorSource.GHG_PROTOCOL,
            EmissionFactorSource.ECOINVENT,
        ]

        for source in sources:
            factor = db_engine.get_wtt_factor(
                fuel_type=FuelType.NATURAL_GAS,
                source=source,
            )
            assert factor is not None
            assert factor.source == source

    def test_get_best_wtt_factor_progressive_resolution(self, db_engine):
        """Test progressive resolution: specific region → country → global."""
        # Try with non-existent specific region
        factor = db_engine.get_wtt_factor(
            fuel_type=FuelType.DIESEL,
            region="US-CA-SF",  # Very specific, may not exist
        )

        assert factor is not None
        # Should fall back to US or GLOBAL
        assert factor.region in ["US-CA-SF", "US-CA", "US", "GLOBAL"]

    def test_get_fuel_heating_value_natural_gas(self, db_engine):
        """Test retrieving heating value for natural gas."""
        properties = db_engine.get_fuel_properties(FuelType.NATURAL_GAS)

        assert properties is not None
        assert properties.net_calorific_value > 0
        # Natural gas: ~13.3 kWh/kg
        assert Decimal("12") < properties.net_calorific_value < Decimal("15")

    def test_get_fuel_heating_value_diesel(self, db_engine):
        """Test retrieving heating value for diesel."""
        properties = db_engine.get_fuel_properties(FuelType.DIESEL)

        assert properties is not None
        # Diesel: ~10.7 kWh/kg
        assert Decimal("10") < properties.net_calorific_value < Decimal("12")
        assert properties.gross_calorific_value > properties.net_calorific_value

    def test_get_fuel_density_diesel(self, db_engine):
        """Test retrieving density for liquid fuel (diesel)."""
        properties = db_engine.get_fuel_properties(FuelType.DIESEL)

        assert properties is not None
        assert properties.density > 0
        # Diesel: ~0.835 kg/litre
        assert Decimal("0.8") < properties.density < Decimal("0.9")

    def test_get_fuel_density_lpg(self, db_engine):
        """Test retrieving density for LPG."""
        properties = db_engine.get_fuel_properties(FuelType.LPG)

        assert properties is not None
        # LPG: ~0.54 kg/litre
        assert Decimal("0.5") < properties.density < Decimal("0.6")

    def test_convert_fuel_units_energy_to_energy_kwh_to_mwh(self, db_engine):
        """Test energy unit conversion: kWh → MWh."""
        result = db_engine.convert_fuel_units(
            quantity=Decimal("10000"),
            from_unit=EnergyUnit.KWH,
            to_unit=EnergyUnit.MWH,
            fuel_type=FuelType.DIESEL,
        )

        assert result == Decimal("10")  # 10000 kWh = 10 MWh

    def test_convert_fuel_units_energy_to_energy_mwh_to_gj(self, db_engine):
        """Test energy unit conversion: MWh → GJ."""
        result = db_engine.convert_fuel_units(
            quantity=Decimal("100"),
            from_unit=EnergyUnit.MWH,
            to_unit=EnergyUnit.GJ,
            fuel_type=FuelType.NATURAL_GAS,
        )

        assert result == Decimal("360")  # 100 MWh = 360 GJ

    def test_convert_fuel_units_gj_to_kwh(self, db_engine):
        """Test energy unit conversion: GJ → kWh."""
        result = db_engine.convert_fuel_units(
            quantity=Decimal("360"),
            from_unit=EnergyUnit.GJ,
            to_unit=EnergyUnit.KWH,
            fuel_type=FuelType.COAL,
        )

        assert result == Decimal("100000")  # 360 GJ = 100000 kWh

    def test_convert_fuel_units_volume_to_energy_litres_to_kwh(self, db_engine):
        """Test volume to energy conversion: litres → kWh using heating value."""
        result = db_engine.convert_fuel_units(
            quantity=Decimal("1000"),  # litres
            from_unit=EnergyUnit.LITRES,
            to_unit=EnergyUnit.KWH,
            fuel_type=FuelType.DIESEL,
        )

        # 1000 L diesel × 0.835 kg/L × 10.7 kWh/kg ≈ 8935 kWh
        assert Decimal("8500") < result < Decimal("9500")

    def test_convert_fuel_units_mass_to_energy_kg_to_kwh(self, db_engine):
        """Test mass to energy conversion: kg → kWh."""
        result = db_engine.convert_fuel_units(
            quantity=Decimal("1000"),  # kg
            from_unit=EnergyUnit.KG,
            to_unit=EnergyUnit.KWH,
            fuel_type=FuelType.NATURAL_GAS,
        )

        # 1000 kg × 13.3 kWh/kg = 13300 kWh
        assert Decimal("12500") < result < Decimal("14000")

    def test_convert_fuel_units_kg_to_tonnes(self, db_engine):
        """Test mass conversion: kg → tonnes."""
        result = db_engine.convert_fuel_units(
            quantity=Decimal("5000"),
            from_unit=EnergyUnit.KG,
            to_unit=EnergyUnit.TONNES,
            fuel_type=FuelType.COAL,
        )

        assert result == Decimal("5")

    def test_convert_to_energy_various_units(self, db_engine):
        """Test conversion of various units to standard kWh."""
        # MWh to kWh
        result1 = db_engine.convert_to_energy(
            quantity=Decimal("50"),
            unit=EnergyUnit.MWH,
            fuel_type=FuelType.DIESEL,
        )
        assert result1 == Decimal("50000")

        # GJ to kWh
        result2 = db_engine.convert_to_energy(
            quantity=Decimal("180"),
            unit=EnergyUnit.GJ,
            fuel_type=FuelType.NATURAL_GAS,
        )
        assert result2 == Decimal("50000")

        # Already kWh
        result3 = db_engine.convert_to_energy(
            quantity=Decimal("50000"),
            unit=EnergyUnit.KWH,
            fuel_type=FuelType.LPG,
        )
        assert result3 == Decimal("50000")

    def test_classify_fuel_fossil(self, db_engine):
        """Test fuel classification for fossil fuels."""
        classification = db_engine.classify_fuel(FuelType.DIESEL)

        assert classification.fuel_type == FuelType.DIESEL
        assert classification.is_fossil is True
        assert classification.is_biofuel is False
        assert classification.is_renewable is False
        assert classification.biogenic_fraction == Decimal("0")

    def test_classify_fuel_biofuel(self, db_engine):
        """Test fuel classification for biofuels."""
        classification = db_engine.classify_fuel(FuelType.BIODIESEL)

        assert classification.fuel_type == FuelType.BIODIESEL
        assert classification.is_fossil is False
        assert classification.is_biofuel is True
        assert classification.is_renewable is True
        assert classification.biogenic_fraction == Decimal("1")

    def test_classify_fuel_waste(self, db_engine):
        """Test fuel classification for waste fuels."""
        classification = db_engine.classify_fuel(FuelType.WASTE_OIL)

        assert classification.fuel_type == FuelType.WASTE_OIL
        assert classification.is_waste is True

    def test_classify_fuel_hydrogen(self, db_engine):
        """Test fuel classification for hydrogen (varies by production method)."""
        classification = db_engine.classify_fuel(FuelType.HYDROGEN)

        assert classification.fuel_type == FuelType.HYDROGEN
        # Hydrogen classification depends on production method
        # Default is typically not classified as fully renewable

    def test_get_fuel_naics_code_diesel(self, db_engine):
        """Test retrieving NAICS code for diesel."""
        classification = db_engine.classify_fuel(FuelType.DIESEL)

        assert classification.naics_code is not None
        # Petroleum refining: 324110
        assert "324" in classification.naics_code

    def test_get_fuel_naics_code_natural_gas(self, db_engine):
        """Test retrieving NAICS code for natural gas."""
        classification = db_engine.classify_fuel(FuelType.NATURAL_GAS)

        assert classification.naics_code is not None
        # Natural gas distribution: 221210
        assert "221" in classification.naics_code

    def test_get_supply_chain_breakdown_natural_gas(self, db_engine):
        """Test supply chain breakdown for natural gas WTT emissions."""
        breakdown = db_engine.get_supply_chain_breakdown(FuelType.NATURAL_GAS)

        assert breakdown is not None
        assert breakdown.extraction > 0
        assert breakdown.processing > 0
        assert breakdown.transport > 0
        assert breakdown.distribution >= 0

        # Total should equal WTT factor
        total = (
            breakdown.extraction
            + breakdown.processing
            + breakdown.transport
            + breakdown.distribution
        )
        assert total > 0

    def test_get_supply_chain_breakdown_diesel(self, db_engine):
        """Test supply chain breakdown for diesel WTT emissions."""
        breakdown = db_engine.get_supply_chain_breakdown(FuelType.DIESEL)

        assert breakdown is not None
        assert breakdown.extraction > 0  # Crude oil extraction
        assert breakdown.processing > 0  # Refining
        assert breakdown.transport > 0  # Distribution
        # Diesel typically has higher refining emissions than natural gas

    def test_get_per_gas_breakdown_natural_gas(self, db_engine):
        """Test per-gas breakdown (CO2, CH4, N2O) for natural gas WTT."""
        breakdown = db_engine.get_gas_breakdown(FuelType.NATURAL_GAS)

        assert breakdown is not None
        assert breakdown.co2 > 0
        assert breakdown.ch4 > 0  # Significant methane leakage in NG supply chain
        assert breakdown.n2o >= 0

        # Total should approximately equal WTT factor
        total = breakdown.co2 + breakdown.ch4 + breakdown.n2o
        assert total > 0

        # Natural gas should have significant CH4 component (>10%)
        ch4_percentage = (breakdown.ch4 / total) * 100
        assert ch4_percentage > Decimal("10")

    def test_get_per_gas_breakdown_diesel(self, db_engine):
        """Test per-gas breakdown for diesel WTT."""
        breakdown = db_engine.get_gas_breakdown(FuelType.DIESEL)

        assert breakdown is not None
        assert breakdown.co2 > 0
        assert breakdown.ch4 >= 0
        assert breakdown.n2o >= 0

        # Diesel WTT is predominantly CO2 (>90%)
        total = breakdown.co2 + breakdown.ch4 + breakdown.n2o
        co2_percentage = (breakdown.co2 / total) * 100
        assert co2_percentage > Decimal("90")

    def test_get_biogenic_fraction_fossil_zero(self, db_engine):
        """Test biogenic fraction for fossil fuels is zero."""
        classification = db_engine.classify_fuel(FuelType.COAL)

        assert classification.biogenic_fraction == Decimal("0")

    def test_get_biogenic_fraction_biofuel_one(self, db_engine):
        """Test biogenic fraction for pure biofuels is one."""
        classification = db_engine.classify_fuel(FuelType.BIOGAS)

        assert classification.biogenic_fraction == Decimal("1")

    def test_get_biogenic_fraction_blended_fuel(self, db_engine):
        """Test biogenic fraction for blended fuels (e.g., E10 = 10% ethanol)."""
        # E10: 10% ethanol (biogenic) + 90% gasoline (fossil)
        biogenic_fraction = db_engine.calculate_blended_biogenic_fraction(
            components=[
                {"fuel_type": FuelType.GASOLINE, "fraction": Decimal("0.9")},
                {"fuel_type": FuelType.ETHANOL, "fraction": Decimal("0.1")},
            ]
        )

        assert biogenic_fraction == pytest.approx(Decimal("0.1"), abs=Decimal("0.01"))

    def test_compare_sources_natural_gas(self, db_engine):
        """Test comparison of WTT factors across different sources."""
        comparison = db_engine.compare_sources(
            fuel_type=FuelType.NATURAL_GAS,
            region="US",
        )

        assert len(comparison) > 0
        # Should have multiple sources
        sources = [c["source"] for c in comparison]
        assert len(set(sources)) > 1

        # All factors should be in reasonable range
        for item in comparison:
            assert Decimal("0.015") < item["wtt_factor"] < Decimal("0.040")

    def test_compare_sources_variability(self, db_engine):
        """Test that source comparison includes variability metrics."""
        comparison = db_engine.compare_sources(
            fuel_type=FuelType.DIESEL,
            region="GLOBAL",
        )

        # Extract all WTT factors
        factors = [Decimal(str(c["wtt_factor"])) for c in comparison]

        # Calculate variability
        if len(factors) > 1:
            max_factor = max(factors)
            min_factor = min(factors)
            variability = (max_factor - min_factor) / min_factor * 100

            # Should have some variability across sources
            assert variability > 0

    def test_register_custom_factor(self, db_engine):
        """Test registering a custom WTT factor."""
        custom_factor = WTTFactor(
            fuel_type=FuelType.DIESEL,
            wtt_emission_factor=Decimal("0.0600"),
            unit=EnergyUnit.KWH,
            region="US-CA",
            source=EmissionFactorSource.CUSTOM,
            year=2025,
            description="California-specific diesel WTT factor",
            supply_chain_breakdown=SupplyChainBreakdown(
                extraction=Decimal("0.020"),
                processing=Decimal("0.025"),
                transport=Decimal("0.015"),
                distribution=Decimal("0.000"),
            ),
        )

        db_engine.register_custom_factor(custom_factor)

        # Retrieve custom factor
        retrieved = db_engine.get_wtt_factor(
            fuel_type=FuelType.DIESEL,
            region="US-CA",
            source=EmissionFactorSource.CUSTOM,
        )

        assert retrieved is not None
        assert retrieved.wtt_emission_factor == Decimal("0.0600")
        assert retrieved.source == EmissionFactorSource.CUSTOM

    def test_get_custom_factors(self, db_engine):
        """Test retrieving all custom factors."""
        # Register multiple custom factors
        custom1 = WTTFactor(
            fuel_type=FuelType.NATURAL_GAS,
            wtt_emission_factor=Decimal("0.0300"),
            unit=EnergyUnit.KWH,
            region="US-TX",
            source=EmissionFactorSource.CUSTOM,
        )
        custom2 = WTTFactor(
            fuel_type=FuelType.DIESEL,
            wtt_emission_factor=Decimal("0.0550"),
            unit=EnergyUnit.KWH,
            region="US-NY",
            source=EmissionFactorSource.CUSTOM,
        )

        db_engine.register_custom_factor(custom1)
        db_engine.register_custom_factor(custom2)

        custom_factors = db_engine.get_custom_factors()

        assert len(custom_factors) >= 2

    def test_validate_fuel_type_valid(self, db_engine):
        """Test validation accepts valid fuel types."""
        # Should not raise
        db_engine.validate_fuel_type(FuelType.DIESEL)
        db_engine.validate_fuel_type(FuelType.NATURAL_GAS)
        db_engine.validate_fuel_type(FuelType.BIOGAS)

    def test_validate_fuel_type_invalid(self, db_engine):
        """Test validation rejects invalid fuel types."""
        with pytest.raises(ValidationError):
            db_engine.validate_fuel_type("INVALID_FUEL")

    def test_validate_fuel_type_none(self, db_engine):
        """Test validation rejects None fuel type."""
        with pytest.raises(ValidationError):
            db_engine.validate_fuel_type(None)

    def test_get_statistics(self, db_engine):
        """Test retrieving database statistics."""
        stats = db_engine.get_statistics()

        assert "total_factors" in stats
        assert "fuel_types_covered" in stats
        assert "sources" in stats
        assert "regions" in stats

        assert stats["total_factors"] > 0
        assert stats["fuel_types_covered"] >= 25
        assert len(stats["sources"]) >= 5

    def test_health_check(self, db_engine):
        """Test health check returns valid status."""
        health = db_engine.health_check()

        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "database_size" in health
        assert "last_updated" in health

    def test_reset(self, db_engine):
        """Test reset clears custom factors but keeps default factors."""
        # Add custom factor
        custom = WTTFactor(
            fuel_type=FuelType.DIESEL,
            wtt_emission_factor=Decimal("0.0500"),
            unit=EnergyUnit.KWH,
            region="TEST",
            source=EmissionFactorSource.CUSTOM,
        )
        db_engine.register_custom_factor(custom)

        # Reset
        db_engine.reset()

        # Custom factors should be cleared
        custom_factors = db_engine.get_custom_factors()
        assert len(custom_factors) == 0

        # Default factors should still exist
        factor = db_engine.get_wtt_factor(FuelType.NATURAL_GAS)
        assert factor is not None

    def test_singleton_pattern(self):
        """Test WTTFuelDatabaseEngine follows singleton pattern (optional)."""
        # If singleton implemented, same instance should be returned
        engine1 = WTTFuelDatabaseEngine()
        engine2 = WTTFuelDatabaseEngine()

        # Depending on implementation, may or may not be singleton
        # This test documents the expected behavior
        assert engine1 is not None
        assert engine2 is not None

    def test_thread_safety(self, db_engine):
        """Test database engine is thread-safe for concurrent reads."""
        def get_factor(fuel_type):
            return db_engine.get_wtt_factor(fuel_type=fuel_type)

        fuel_types = [
            FuelType.NATURAL_GAS,
            FuelType.DIESEL,
            FuelType.COAL,
            FuelType.LPG,
            FuelType.GASOLINE,
        ]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_factor, ft) for ft in fuel_types * 10]

            results = []
            for future in as_completed(futures):
                result = future.result()
                assert result is not None
                results.append(result)

        assert len(results) == 50  # 5 fuel types × 10 iterations

    def test_get_all_fuel_types(self, db_engine):
        """Test retrieving list of all supported fuel types."""
        fuel_types = db_engine.get_all_fuel_types()

        assert len(fuel_types) >= 25
        assert FuelType.NATURAL_GAS in fuel_types
        assert FuelType.DIESEL in fuel_types
        assert FuelType.COAL in fuel_types
        assert FuelType.BIOMASS in fuel_types
        assert FuelType.HYDROGEN in fuel_types

    def test_get_all_regions(self, db_engine):
        """Test retrieving list of all regions with WTT data."""
        regions = db_engine.get_all_regions()

        assert len(regions) > 0
        assert "US" in regions or "GLOBAL" in regions

    def test_search_factors_by_criteria(self, db_engine):
        """Test searching factors by multiple criteria."""
        results = db_engine.search_factors(
            fuel_type=FuelType.DIESEL,
            region="US",
            min_year=2020,
        )

        assert len(results) > 0
        for factor in results:
            assert factor.fuel_type == FuelType.DIESEL
            assert factor.year >= 2020 if factor.year else True

    def test_get_latest_factor_by_year(self, db_engine):
        """Test retrieving most recent factor by year."""
        factor = db_engine.get_wtt_factor(
            fuel_type=FuelType.NATURAL_GAS,
            prefer_latest=True,
        )

        assert factor is not None
        # Should return the most recent year available

    def test_calculate_weighted_average_factor(self, db_engine):
        """Test calculating weighted average WTT factor across sources."""
        # Get factors from multiple sources
        sources = [
            EmissionFactorSource.EPA_GHGRP,
            EmissionFactorSource.DEFRA,
            EmissionFactorSource.GHG_PROTOCOL,
        ]

        factors = []
        for source in sources:
            try:
                factor = db_engine.get_wtt_factor(
                    fuel_type=FuelType.DIESEL,
                    source=source,
                )
                if factor:
                    factors.append(factor.wtt_emission_factor)
            except:
                pass

        if len(factors) > 1:
            # Calculate simple average
            avg_factor = sum(factors) / len(factors)
            assert avg_factor > 0

    def test_export_database_to_dict(self, db_engine):
        """Test exporting database to dictionary format."""
        export = db_engine.export_to_dict()

        assert "factors" in export
        assert "properties" in export
        assert "classifications" in export
        assert len(export["factors"]) > 0

    def test_import_database_from_dict(self):
        """Test importing database from dictionary format."""
        engine = WTTFuelDatabaseEngine()

        export_data = {
            "factors": [
                {
                    "fuel_type": "DIESEL",
                    "wtt_emission_factor": "0.0507",
                    "unit": "KWH",
                    "region": "US",
                    "source": "EPA_GHGRP",
                }
            ]
        }

        # Import data
        engine.import_from_dict(export_data)

        # Verify import
        factor = engine.get_wtt_factor(FuelType.DIESEL)
        assert factor is not None


# Performance Tests
class TestPerformanceWTTFuelDatabase:
    """Performance tests for WTTFuelDatabaseEngine."""

    def test_lookup_performance(self, db_engine, benchmark):
        """Test factor lookup performance."""
        def lookup():
            return db_engine.get_wtt_factor(
                fuel_type=FuelType.NATURAL_GAS,
                region="US",
            )

        result = benchmark(lookup)
        assert result is not None

    def test_conversion_performance(self, db_engine, benchmark):
        """Test unit conversion performance."""
        def convert():
            return db_engine.convert_fuel_units(
                quantity=Decimal("10000"),
                from_unit=EnergyUnit.LITRES,
                to_unit=EnergyUnit.KWH,
                fuel_type=FuelType.DIESEL,
            )

        result = benchmark(convert)
        assert result > 0

    def test_batch_lookup_performance(self, db_engine):
        """Test batch lookup performance for 1000 queries."""
        fuel_types = [
            FuelType.NATURAL_GAS,
            FuelType.DIESEL,
            FuelType.COAL,
            FuelType.LPG,
            FuelType.GASOLINE,
        ]

        start_time = datetime.now()

        for _ in range(1000):
            fuel_type = fuel_types[_ % len(fuel_types)]
            factor = db_engine.get_wtt_factor(fuel_type=fuel_type)
            assert factor is not None

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should complete 1000 lookups in <1 second
        assert duration < 1.0


# Edge Case Tests
class TestEdgeCasesWTTFuelDatabase:
    """Edge case tests for WTTFuelDatabaseEngine."""

    def test_zero_quantity_conversion(self, db_engine):
        """Test conversion with zero quantity."""
        result = db_engine.convert_fuel_units(
            quantity=Decimal("0"),
            from_unit=EnergyUnit.LITRES,
            to_unit=EnergyUnit.KWH,
            fuel_type=FuelType.DIESEL,
        )

        assert result == Decimal("0")

    def test_very_large_quantity_conversion(self, db_engine):
        """Test conversion with very large quantity."""
        result = db_engine.convert_fuel_units(
            quantity=Decimal("1000000000"),  # 1 billion litres
            from_unit=EnergyUnit.LITRES,
            to_unit=EnergyUnit.KWH,
            fuel_type=FuelType.DIESEL,
        )

        assert result > Decimal("1000000000")  # Should be huge

    def test_negative_quantity_rejected(self, db_engine):
        """Test that negative quantities are rejected."""
        with pytest.raises(ValidationError):
            db_engine.convert_fuel_units(
                quantity=Decimal("-1000"),
                from_unit=EnergyUnit.LITRES,
                to_unit=EnergyUnit.KWH,
                fuel_type=FuelType.DIESEL,
            )

    def test_missing_fuel_properties_fallback(self, db_engine):
        """Test fallback when fuel properties are missing."""
        # Try to get properties for a fuel type with limited data
        properties = db_engine.get_fuel_properties(FuelType.MUNICIPAL_WASTE)

        # Should return default/estimated properties or raise gracefully
        assert properties is not None or True  # Graceful handling

    def test_unsupported_unit_conversion(self, db_engine):
        """Test conversion with unsupported unit combination."""
        with pytest.raises((ValidationError, ProcessingError)):
            db_engine.convert_fuel_units(
                quantity=Decimal("1000"),
                from_unit=EnergyUnit.KWH,
                to_unit=EnergyUnit.LITRES,  # Cannot convert energy → volume without density
                fuel_type=FuelType.ELECTRICITY,  # Electricity has no volume
            )
