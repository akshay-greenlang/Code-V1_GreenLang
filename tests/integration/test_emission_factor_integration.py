"""
Integration Tests for Emission Factor Database

These tests verify the complete emission factor system integration:
- Database population from YAML files
- EmissionFactorClient SDK
- FactorBrokerAdapter backward compatibility
- Calculator agent integrations
- Performance and caching

Run with: pytest tests/integration/test_emission_factor_integration.py -v

Author: GreenLang Backend Team
Version: 1.0.0
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from greenlang.sdk.emission_factor_client import (
    EmissionFactorClient,
    EmissionFactorNotFoundError,
    UnitNotAvailableError,
    DatabaseConnectionError
)
from greenlang.adapters.factor_broker_adapter import create_factor_broker
from greenlang.models.emission_factor import FactorSearchCriteria, DataQualityTier


class TestEmissionFactorDatabase:
    """Test emission factor database integrity."""

    def test_database_exists(self):
        """Test that database file exists."""
        # Try to create client
        with EmissionFactorClient() as client:
            assert client is not None

    def test_database_has_factors(self):
        """Test that database contains emission factors."""
        with EmissionFactorClient() as client:
            stats = client.get_statistics()
            assert stats['total_factors'] > 0, "Database should contain emission factors"

    def test_database_categories(self):
        """Test that database has expected categories."""
        expected_categories = ["fuels", "grids", "processes", "business_travel"]

        with EmissionFactorClient() as client:
            stats = client.get_statistics()
            categories = stats['by_category'].keys()

            for expected in expected_categories:
                assert expected in categories, f"Missing category: {expected}"

    def test_database_scopes(self):
        """Test that database has factors for all scopes."""
        expected_scopes = ["Scope 1", "Scope 2", "Scope 3"]

        with EmissionFactorClient() as client:
            stats = client.get_statistics()
            scopes = [s for s in stats['by_scope'].keys() if s]  # Filter None

            # At least some scopes should be present
            assert len(scopes) > 0, "Database should have scope classifications"


class TestEmissionFactorLookup:
    """Test emission factor lookup functionality."""

    def test_get_factor_by_id(self):
        """Test getting factor by exact ID."""
        with EmissionFactorClient() as client:
            factor = client.get_factor("fuels_diesel")

            assert factor is not None
            assert factor.factor_id == "fuels_diesel"
            assert factor.emission_factor_kg_co2e > 0

    def test_get_factor_by_name(self):
        """Test searching factors by name."""
        with EmissionFactorClient() as client:
            factors = client.get_factor_by_name("diesel")

            assert len(factors) > 0
            assert any("diesel" in f.name.lower() for f in factors)

    def test_get_grid_factor(self):
        """Test getting grid emission factor."""
        with EmissionFactorClient() as client:
            factor = client.get_grid_factor("California")

            assert factor is not None
            assert factor.category == "grids"
            assert factor.emission_factor_kg_co2e > 0

    def test_get_fuel_factor(self):
        """Test getting fuel emission factor."""
        with EmissionFactorClient() as client:
            factor = client.get_fuel_factor("diesel", unit="gallon")

            assert factor is not None
            assert factor.category == "fuels"

    def test_get_nonexistent_factor(self):
        """Test that missing factor raises error."""
        with EmissionFactorClient() as client:
            with pytest.raises(EmissionFactorNotFoundError):
                client.get_factor("nonexistent_factor_12345")

    def test_get_by_category(self):
        """Test getting factors by category."""
        with EmissionFactorClient() as client:
            factors = client.get_by_category("fuels")

            assert len(factors) > 0
            assert all(f.category == "fuels" for f in factors)

    def test_get_by_scope(self):
        """Test getting factors by scope."""
        with EmissionFactorClient() as client:
            factors = client.get_by_scope("Scope 1")

            assert len(factors) > 0
            assert all(f.scope == "Scope 1" for f in factors)


class TestEmissionCalculations:
    """Test emission calculation functionality."""

    def test_calculate_emissions_basic(self):
        """Test basic emissions calculation."""
        with EmissionFactorClient() as client:
            result = client.calculate_emissions(
                factor_id="fuels_diesel",
                activity_amount=100.0,
                activity_unit="gallon"
            )

            assert result.emissions_kg_co2e > 0
            assert result.activity_amount == 100.0
            assert result.activity_unit == "gallon"
            assert result.audit_trail is not None

    def test_calculate_emissions_provenance(self):
        """Test that calculation includes provenance."""
        with EmissionFactorClient() as client:
            result = client.calculate_emissions(
                factor_id="fuels_diesel",
                activity_amount=100.0,
                activity_unit="gallon"
            )

            assert result.factor_used is not None
            assert result.factor_used.source.source_uri is not None
            assert result.calculation_timestamp is not None

    def test_calculate_emissions_reproducible(self):
        """Test that calculations are reproducible."""
        with EmissionFactorClient() as client:
            result1 = client.calculate_emissions(
                factor_id="fuels_diesel",
                activity_amount=100.0,
                activity_unit="gallon"
            )

            result2 = client.calculate_emissions(
                factor_id="fuels_diesel",
                activity_amount=100.0,
                activity_unit="gallon"
            )

            # Same inputs should give same emissions (deterministic)
            assert result1.emissions_kg_co2e == result2.emissions_kg_co2e
            assert result1.factor_value_applied == result2.factor_value_applied

    def test_calculate_emissions_zero_activity(self):
        """Test calculation with zero activity."""
        with EmissionFactorClient() as client:
            result = client.calculate_emissions(
                factor_id="fuels_diesel",
                activity_amount=0.0,
                activity_unit="gallon"
            )

            assert result.emissions_kg_co2e == 0.0

    def test_calculate_emissions_negative_activity(self):
        """Test that negative activity raises error."""
        with EmissionFactorClient() as client:
            with pytest.raises(ValueError):
                client.calculate_emissions(
                    factor_id="fuels_diesel",
                    activity_amount=-100.0,
                    activity_unit="gallon"
                )

    def test_calculate_emissions_invalid_unit(self):
        """Test calculation with invalid unit raises error."""
        with EmissionFactorClient() as client:
            with pytest.raises(UnitNotAvailableError):
                client.calculate_emissions(
                    factor_id="fuels_diesel",
                    activity_amount=100.0,
                    activity_unit="invalid_unit"
                )


class TestUnitConversions:
    """Test unit conversion functionality."""

    def test_multiple_units_available(self):
        """Test that factors have multiple units."""
        with EmissionFactorClient() as client:
            factor = client.get_factor("fuels_diesel")

            # Diesel should have multiple units
            assert len(factor.additional_units) > 0

    def test_get_factor_for_unit(self):
        """Test getting emission factor for specific unit."""
        with EmissionFactorClient() as client:
            factor = client.get_factor("fuels_diesel")

            # Test different units
            gallon_ef = factor.get_factor_for_unit("gallon")
            liter_ef = factor.get_factor_for_unit("liter")

            assert gallon_ef > 0
            assert liter_ef > 0
            # Gallon should be higher than liter (1 gallon ~ 3.78 liters)
            assert gallon_ef > liter_ef

    def test_invalid_unit_raises_error(self):
        """Test that invalid unit raises error."""
        with EmissionFactorClient() as client:
            factor = client.get_factor("fuels_diesel")

            with pytest.raises(ValueError):
                factor.get_factor_for_unit("invalid_unit_xyz")


class TestFactorBrokerAdapter:
    """Test backward compatibility adapter."""

    def test_create_broker(self):
        """Test creating broker adapter."""
        with create_factor_broker() as broker:
            assert broker is not None

    def test_broker_get_factor(self):
        """Test broker get_factor method."""
        with create_factor_broker() as broker:
            ef_value = broker.get_factor("diesel", unit="gallons")

            assert ef_value > 0
            assert isinstance(ef_value, float)

    def test_broker_calculate(self):
        """Test broker calculate method."""
        with create_factor_broker() as broker:
            emissions = broker.calculate("diesel", 100.0, "gallons")

            assert emissions > 0
            assert isinstance(emissions, float)

    def test_broker_calculate_detailed(self):
        """Test broker calculate_detailed method."""
        with create_factor_broker() as broker:
            result = broker.calculate_detailed("diesel", 100.0, "gallons")

            assert result.emissions_kg_co2e > 0
            assert result.audit_trail is not None
            assert result.factor_used is not None

    def test_broker_get_grid_factor(self):
        """Test broker grid factor lookup."""
        with create_factor_broker() as broker:
            ef_value = broker.get_grid_factor("California")

            assert ef_value > 0

    def test_broker_get_fuel_factor(self):
        """Test broker fuel factor lookup."""
        with create_factor_broker() as broker:
            ef_value = broker.get_fuel_factor("diesel", "gallons")

            assert ef_value > 0


class TestPerformance:
    """Test performance and caching."""

    def test_lookup_performance(self):
        """Test that lookups are fast."""
        with EmissionFactorClient() as client:
            start = time.time()
            factor = client.get_factor("fuels_diesel")
            elapsed_ms = (time.time() - start) * 1000

            # Should complete in less than 100ms
            assert elapsed_ms < 100, f"Lookup took {elapsed_ms:.2f}ms (should be < 100ms)"

    def test_calculation_performance(self):
        """Test that calculations are fast."""
        with EmissionFactorClient() as client:
            start = time.time()
            result = client.calculate_emissions(
                factor_id="fuels_diesel",
                activity_amount=100.0,
                activity_unit="gallon"
            )
            elapsed_ms = (time.time() - start) * 1000

            # Should complete in less than 100ms
            assert elapsed_ms < 100, f"Calculation took {elapsed_ms:.2f}ms (should be < 100ms)"

    def test_batch_performance(self):
        """Test batch operations performance."""
        factor_ids = [
            "fuels_diesel",
            "fuels_gasoline_motor",
            "fuels_natural_gas",
            "grids_us_national",
            "grids_us_wecc_ca"
        ]

        with EmissionFactorClient() as client:
            start = time.time()
            for fid in factor_ids:
                client.get_factor(fid)
            elapsed_ms = (time.time() - start) * 1000

            # Average per factor should be < 50ms
            avg_ms = elapsed_ms / len(factor_ids)
            assert avg_ms < 50, f"Average lookup: {avg_ms:.2f}ms (should be < 50ms)"


class TestDataQuality:
    """Test data quality and metadata."""

    def test_factor_has_source(self):
        """Test that factors have source information."""
        with EmissionFactorClient() as client:
            factor = client.get_factor("fuels_diesel")

            assert factor.source.source_org is not None
            assert factor.source.source_uri is not None

    def test_factor_has_last_updated(self):
        """Test that factors have last_updated date."""
        with EmissionFactorClient() as client:
            factor = client.get_factor("fuels_diesel")

            assert factor.last_updated is not None

    def test_factor_data_quality(self):
        """Test that factors have data quality information."""
        with EmissionFactorClient() as client:
            factor = client.get_factor("fuels_diesel")

            assert factor.data_quality is not None
            assert factor.data_quality.tier in [t.value for t in DataQualityTier]

    def test_stale_factor_warning(self):
        """Test that stale factors are detected."""
        with EmissionFactorClient() as client:
            # Get an old factor (if any exist)
            stats = client.get_statistics()

            if stats['stale_factors'] > 0:
                # This is expected and should log a warning
                pass


class TestAuditTrail:
    """Test audit trail and provenance."""

    def test_calculation_logged(self):
        """Test that calculations are logged to audit table."""
        with EmissionFactorClient() as client:
            # Get initial calculation count
            stats_before = client.get_statistics()
            count_before = stats_before['total_calculations']

            # Perform calculation
            result = client.calculate_emissions(
                factor_id="fuels_diesel",
                activity_amount=100.0,
                activity_unit="gallon"
            )

            # Get updated count
            stats_after = client.get_statistics()
            count_after = stats_after['total_calculations']

            # Count should have increased
            assert count_after > count_before

    def test_audit_hash_unique(self):
        """Test that audit hashes are unique."""
        with EmissionFactorClient() as client:
            result1 = client.calculate_emissions(
                factor_id="fuels_diesel",
                activity_amount=100.0,
                activity_unit="gallon"
            )

            # Wait a moment to ensure different timestamp
            time.sleep(0.01)

            result2 = client.calculate_emissions(
                factor_id="fuels_diesel",
                activity_amount=100.0,
                activity_unit="gallon"
            )

            # Different timestamps should give different hashes
            # (even though emissions are the same)
            assert result1.audit_trail != result2.audit_trail


# Performance benchmarks (run separately)
@pytest.mark.benchmark
class TestBenchmarks:
    """Performance benchmarks (run with pytest --benchmark)."""

    def test_benchmark_factor_lookup(self, benchmark):
        """Benchmark factor lookup performance."""
        def lookup():
            with EmissionFactorClient() as client:
                return client.get_factor("fuels_diesel")

        result = benchmark(lookup)
        assert result is not None

    def test_benchmark_calculation(self, benchmark):
        """Benchmark emissions calculation performance."""
        def calculate():
            with EmissionFactorClient() as client:
                return client.calculate_emissions(
                    factor_id="fuels_diesel",
                    activity_amount=100.0,
                    activity_unit="gallon"
                )

        result = benchmark(calculate)
        assert result.emissions_kg_co2e > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
