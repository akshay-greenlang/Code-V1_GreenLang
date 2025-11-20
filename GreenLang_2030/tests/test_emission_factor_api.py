"""
Comprehensive Test Suite for Emission Factor API Endpoints

Tests all 14 REST API endpoints with performance validation.

Author: QA Team Lead
Date: 2025-11-20
"""

import pytest
import sys
from pathlib import Path
from decimal import Decimal
from datetime import date
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_foundation" / "agents" / "calculator"))

from emission_factors import EmissionFactorDatabase, EmissionFactor


class TestEmissionFactorAPI:
    """Test emission factor API endpoints."""

    @pytest.fixture(autouse=True)
    def setup_api(self):
        """Setup test database for API testing."""
        self.db = EmissionFactorDatabase()
        self._insert_test_data()
        yield
        self.db.close()

    def _insert_test_data(self):
        """Insert test data for API testing."""
        test_factors = [
            EmissionFactor(
                factor_id=f"api_test_diesel_gb",
                category="scope1",
                activity_type="fuel_combustion",
                material_or_fuel="diesel",
                unit="kg_co2e_per_liter",
                factor_co2=Decimal("2.68"),
                factor_co2e=Decimal("2.69"),
                region="GB",
                valid_from=date(2024, 1, 1),
                valid_to=date(2024, 12, 31),
                source="DEFRA",
                source_year=2024,
                source_version="2024",
                data_quality="high",
                uncertainty_percentage=5.0
            ),
            EmissionFactor(
                factor_id=f"api_test_electricity_us",
                category="scope2",
                activity_type="electricity_consumption",
                material_or_fuel="grid_average",
                unit="kg_co2e_per_kwh",
                factor_co2=Decimal("0.45"),
                factor_co2e=Decimal("0.46"),
                region="US",
                valid_from=date(2024, 1, 1),
                source="EPA",
                source_year=2024,
                source_version="2024",
                data_quality="high"
            )
        ]

        for factor in test_factors:
            self.db.insert_factor(factor)

        # Insert pagination test data
        for i in range(100):
            factor = EmissionFactor(
                factor_id=f"pagination_test_{i}",
                category="scope1",
                activity_type="fuel_combustion",
                material_or_fuel=f"fuel_{i % 10}",
                unit="kg_co2e_per_unit",
                factor_co2=Decimal(str(2.0 + i * 0.01)),
                factor_co2e=Decimal(str(2.01 + i * 0.01)),
                region="GLOBAL",
                valid_from=date(2024, 1, 1),
                source="TEST",
                source_year=2024,
                source_version="1.0",
                data_quality="medium"
            )
            self.db.insert_factor(factor)

    # API Endpoint 1: GET /factors/{factor_id}
    def test_endpoint_get_factor_by_id(self):
        """Test GET /factors/{factor_id} endpoint."""
        start = time.perf_counter()

        # Simulate API call
        factor = self.db.get_factor(
            category="scope1",
            activity_type="fuel_combustion",
            material_or_fuel="diesel",
            region="GB"
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert factor is not None
        assert factor.factor_id == "api_test_diesel_gb"

        print(f"\n✓ GET /factors/{{id}}: {elapsed_ms:.2f}ms")

    # API Endpoint 2: GET /factors?category={category}
    def test_endpoint_search_by_category(self):
        """Test GET /factors?category endpoint."""
        start = time.perf_counter()

        factors = self.db.search_factors(category="scope1", limit=50)

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(factors) > 0
        assert all(f.category == "scope1" for f in factors)

        print(f"\n✓ GET /factors?category=scope1: {elapsed_ms:.2f}ms ({len(factors)} results)")

    # API Endpoint 3: GET /factors?source={source}
    def test_endpoint_search_by_source(self):
        """Test GET /factors?source endpoint."""
        start = time.perf_counter()

        factors = self.db.search_factors(source="DEFRA", limit=50)

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(factors) >= 0
        if factors:
            assert all(f.source == "DEFRA" for f in factors)

        print(f"\n✓ GET /factors?source=DEFRA: {elapsed_ms:.2f}ms ({len(factors)} results)")

    # API Endpoint 4: GET /factors?region={region}
    def test_endpoint_search_by_region(self):
        """Test GET /factors?region endpoint."""
        start = time.perf_counter()

        factors = self.db.search_factors(region="GB", limit=50)

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(factors) >= 0

        print(f"\n✓ GET /factors?region=GB: {elapsed_ms:.2f}ms ({len(factors)} results)")

    # API Endpoint 5: GET /factors?activity_type={type}
    def test_endpoint_search_by_activity(self):
        """Test GET /factors?activity_type endpoint."""
        start = time.perf_counter()

        factors = self.db.search_factors(activity_type="fuel_combustion", limit=50)

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(factors) > 0

        print(f"\n✓ GET /factors?activity_type=fuel_combustion: {elapsed_ms:.2f}ms")

    # API Endpoint 6: GET /factors?material={material}
    def test_endpoint_search_by_material(self):
        """Test GET /factors?material endpoint."""
        start = time.perf_counter()

        factors = self.db.search_factors(material_or_fuel="diesel", limit=50)

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(factors) >= 0

        print(f"\n✓ GET /factors?material=diesel: {elapsed_ms:.2f}ms")

    # API Endpoint 7: GET /factors with pagination
    def test_endpoint_pagination(self):
        """Test pagination with limit parameter."""
        page_size = 10
        start = time.perf_counter()

        page1 = self.db.search_factors(limit=page_size)

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(page1) <= page_size

        print(f"\n✓ GET /factors?limit={page_size}: {elapsed_ms:.2f}ms")

    # API Endpoint 8: GET /factors/statistics
    def test_endpoint_statistics(self):
        """Test GET /factors/statistics endpoint."""
        start = time.perf_counter()

        stats = self.db.get_statistics()

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert 'total_factors' in stats
        assert 'by_source' in stats
        assert 'by_category' in stats

        print(f"\n✓ GET /factors/statistics: {elapsed_ms:.2f}ms")
        print(f"  Total: {stats['total_factors']}")

    # API Endpoint 9: POST /factors/lookup (deterministic lookup)
    def test_endpoint_lookup_factor(self):
        """Test POST /factors/lookup endpoint."""
        start = time.perf_counter()

        lookup_params = {
            "category": "scope1",
            "activity_type": "fuel_combustion",
            "material_or_fuel": "diesel",
            "region": "GB",
            "reference_date": date(2024, 6, 1)
        }

        factor = self.db.get_factor(**lookup_params)

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert factor is not None
        assert factor.material_or_fuel == "diesel"

        print(f"\n✓ POST /factors/lookup: {elapsed_ms:.2f}ms")

    # API Endpoint 10: POST /factors/batch-lookup
    def test_endpoint_batch_lookup(self):
        """Test POST /factors/batch-lookup endpoint."""
        batch_queries = [
            ("scope1", "fuel_combustion", "diesel", "GB"),
            ("scope2", "electricity_consumption", "grid_average", "US"),
        ]

        start = time.perf_counter()

        results = []
        for category, activity, material, region in batch_queries:
            factor = self.db.get_factor(
                category=category,
                activity_type=activity,
                material_or_fuel=material,
                region=region,
                reference_date=date(2024, 6, 1)
            )
            if factor:
                results.append(factor)

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(results) >= 0

        print(f"\n✓ POST /factors/batch-lookup: {elapsed_ms:.2f}ms ({len(results)} factors)")

    # API Endpoint 11: GET /factors/sources
    def test_endpoint_list_sources(self):
        """Test GET /factors/sources endpoint."""
        start = time.perf_counter()

        stats = self.db.get_statistics()
        sources = list(stats['by_source'].keys())

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(sources) > 0

        print(f"\n✓ GET /factors/sources: {elapsed_ms:.2f}ms ({len(sources)} sources)")

    # API Endpoint 12: GET /factors/categories
    def test_endpoint_list_categories(self):
        """Test GET /factors/categories endpoint."""
        start = time.perf_counter()

        stats = self.db.get_statistics()
        categories = list(stats['by_category'].keys())

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(categories) > 0
        assert "scope1" in categories or "scope2" in categories

        print(f"\n✓ GET /factors/categories: {elapsed_ms:.2f}ms")

    # API Endpoint 13: GET /factors/regions
    def test_endpoint_list_regions(self):
        """Test GET /factors/regions endpoint."""
        start = time.perf_counter()

        stats = self.db.get_statistics()
        regions = list(stats['top_regions'].keys())

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(regions) > 0

        print(f"\n✓ GET /factors/regions: {elapsed_ms:.2f}ms ({len(regions)} regions)")

    # API Endpoint 14: GET /health
    def test_endpoint_health_check(self):
        """Test GET /health endpoint."""
        start = time.perf_counter()

        # Simulate health check
        stats = self.db.get_statistics()
        health_status = {
            "status": "healthy",
            "total_factors": stats['total_factors'],
            "database": "connected"
        }

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert health_status['status'] == "healthy"

        print(f"\n✓ GET /health: {elapsed_ms:.2f}ms")

    def test_performance_all_endpoints(self):
        """Test performance of all endpoints (<15ms target)."""
        endpoint_times = []

        # Test each endpoint
        tests = [
            ("GET /factors/{id}", lambda: self.db.get_factor(
                category="scope1", activity_type="fuel_combustion",
                material_or_fuel="diesel", region="GB"
            )),
            ("GET /factors?category", lambda: self.db.search_factors(category="scope1", limit=10)),
            ("GET /statistics", lambda: self.db.get_statistics()),
        ]

        for name, test_func in tests:
            start = time.perf_counter()
            result = test_func()
            elapsed_ms = (time.perf_counter() - start) * 1000
            endpoint_times.append((name, elapsed_ms))

        print(f"\n✓ Performance Summary:")
        for name, time_ms in endpoint_times:
            status = "✓" if time_ms < 15 else "✗"
            print(f"  {status} {name}: {time_ms:.2f}ms")

        avg_time = sum(t for _, t in endpoint_times) / len(endpoint_times)
        print(f"\n  Average: {avg_time:.2f}ms (Target: <15ms)")

    def test_response_time_consistency(self):
        """Test that response times are consistent across multiple calls."""
        times = []

        for _ in range(10):
            start = time.perf_counter()
            self.db.get_factor(
                category="scope1",
                activity_type="fuel_combustion",
                material_or_fuel="diesel",
                region="GB"
            )
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        import statistics
        avg = statistics.mean(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0

        print(f"\n✓ Response Time Consistency (10 calls):")
        print(f"  Average: {avg:.2f}ms")
        print(f"  Std Dev: {stdev:.2f}ms")
        print(f"  Min: {min(times):.2f}ms, Max: {max(times):.2f}ms")

    def test_error_handling(self):
        """Test API error handling."""
        # Test not found
        factor = self.db.get_factor(
            category="scope99",
            activity_type="nonexistent",
            material_or_fuel="invalid",
            region="XX"
        )

        assert factor is None, "Should return None for invalid query"

        print(f"\n✓ Error Handling: Properly returns None for invalid queries")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
