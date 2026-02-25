"""
Unit tests for PurchasedGoodsServicesService setup and integration (AGENT-MRV-014).

Tests cover:
- Singleton pattern
- Service methods
- Integration functions
- Health checks
- Operation statistics
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

try:
    from greenlang.purchased_goods_services.setup import (
        PurchasedGoodsServicesService,
        get_service,
        get_router,
        configure_purchased_goods,
    )
    from greenlang.purchased_goods_services.spend_based_calculator import SpendBasedInput
    from greenlang.purchased_goods_services.average_data_calculator import AverageDataInput
    from greenlang.purchased_goods_services.supplier_specific_calculator import SupplierSpecificInput
    from greenlang.purchased_goods_services.purchased_goods_pipeline import PipelineInput
except ImportError:
    pytest.skip("PurchasedGoodsServicesService not available", allow_module_level=True)


class TestPurchasedGoodsServicesServiceSingleton:
    """Test singleton pattern for PurchasedGoodsServicesService."""

    def test_singleton_same_instance(self):
        """Test that get_service returns same instance."""
        service1 = get_service()
        service2 = get_service()
        assert service1 is service2

    def test_singleton_direct_instantiation(self):
        """Test direct instantiation returns singleton."""
        service1 = PurchasedGoodsServicesService()
        service2 = PurchasedGoodsServicesService()
        assert service1 is service2

    def test_singleton_thread_safe(self):
        """Test thread-safe singleton creation."""
        import threading
        instances = []

        def get_instance():
            instances.append(get_service())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)


class TestServiceMethods:
    """Test service methods."""

    def test_calculate_spend_based(self):
        """Test spend-based calculation method."""
        service = get_service()

        input_data = SpendBasedInput(
            spend_amount=Decimal("100000"),
            category="raw_materials",
            reporting_year=2024,
        )

        result = service.calculate_spend_based(input_data)

        assert result is not None
        assert "total_emissions" in result
        assert result["total_emissions"] > 0

    def test_calculate_average_data(self):
        """Test average-data calculation method."""
        service = get_service()

        input_data = AverageDataInput(
            product_id="P001",
            quantity=Decimal("1000"),
            unit="kg",
            product_category="steel",
            reporting_year=2024,
        )

        result = service.calculate_average_data(input_data)

        assert result is not None
        assert "total_emissions" in result

    def test_calculate_supplier_specific(self):
        """Test supplier-specific calculation method."""
        service = get_service()

        from greenlang.purchased_goods_services.supplier_specific_calculator import ProductEmissions

        product = ProductEmissions(
            product_id="P001",
            quantity=Decimal("1000"),
            unit="kg",
            emission_factor=Decimal("2.5"),
            ef_unit="kg_co2e_per_kg",
        )

        result = service.calculate_supplier_specific(product)

        assert result is not None
        assert "total_emissions" in result
        assert result["total_emissions"] == Decimal("2500.0")

    def test_calculate_hybrid(self):
        """Test hybrid calculation method."""
        service = get_service()

        items = [
            {"id": "P001", "spend": Decimal("10000"), "category": "raw_materials"},
            {"id": "P002", "spend": Decimal("8000"), "category": "packaging"},
        ]

        result = service.calculate_hybrid(items)

        assert result is not None
        assert "total_emissions" in result
        assert "coverage" in result

    def test_run_pipeline(self):
        """Test full pipeline execution."""
        service = get_service()

        input_data = {
            "items": [
                {"id": "P001", "spend": Decimal("10000"), "category": "raw_materials"},
                {"id": "P002", "spend": Decimal("5000"), "category": "packaging"},
            ],
            "reporting_year": 2024,
        }

        result = service.run_pipeline(input_data)

        assert result is not None
        assert "total_emissions" in result
        assert "stages" in result
        assert len(result["stages"]) == 10

    def test_lookup_eeio_factor(self):
        """Test EEIO factor lookup."""
        service = get_service()

        factor = service.lookup_eeio_factor(
            category="raw_materials",
            taxonomy="naics",
            year=2024,
        )

        assert factor is not None
        assert isinstance(factor, Decimal)
        assert factor > 0

    def test_lookup_avgdata_factor(self):
        """Test average-data factor lookup."""
        service = get_service()

        factor = service.lookup_avgdata_factor(
            product_category="steel",
            region="US",
            year=2024,
        )

        assert factor is not None
        assert isinstance(factor, Decimal)

    def test_check_compliance(self):
        """Test compliance checking."""
        service = get_service()

        data = {
            "coverage": Decimal("95.0"),
            "dqi": Decimal("1.8"),
            "boundary_complete": True,
            "documentation_complete": True,
            "base_year_set": True,
        }

        result = service.check_compliance(data, frameworks=["ghg_protocol", "csrd_esrs"])

        assert result is not None
        assert "ghg_protocol" in result
        assert "csrd_esrs" in result

    def test_analyze_hotspots(self):
        """Test hot-spot analysis."""
        service = get_service()

        items = [
            {"id": "P001", "emissions": Decimal("5000"), "spend": Decimal("100000")},
            {"id": "P002", "emissions": Decimal("3000"), "spend": Decimal("80000")},
            {"id": "P003", "emissions": Decimal("500"), "spend": Decimal("20000")},
        ]

        result = service.analyze_hotspots(items, top_n=2)

        assert result is not None
        assert len(result["top_emitters"]) == 2
        assert result["top_emitters"][0]["id"] == "P001"

    def test_calculate_coverage(self):
        """Test coverage calculation."""
        service = get_service()

        coverage = service.calculate_coverage(
            covered_spend=Decimal("750000"),
            total_spend=Decimal("1000000"),
        )

        assert coverage == Decimal("75.0")


class TestIntegrationFunctions:
    """Test integration functions."""

    def test_get_service_returns_instance(self):
        """Test get_service returns service instance."""
        service = get_service()

        assert isinstance(service, PurchasedGoodsServicesService)

    def test_get_router_returns_router(self):
        """Test get_router returns API router."""
        router = get_router()

        assert router is not None
        assert hasattr(router, 'routes')

    def test_configure_purchased_goods_with_app(self):
        """Test configure_purchased_goods with FastAPI app."""
        from fastapi import FastAPI

        app = FastAPI()

        configure_purchased_goods(app)

        # Check router was added
        assert len(app.routes) > 0

    def test_configure_purchased_goods_idempotent(self):
        """Test configure_purchased_goods can be called multiple times."""
        from fastapi import FastAPI

        app = FastAPI()

        configure_purchased_goods(app)
        routes_count_1 = len(app.routes)

        configure_purchased_goods(app)
        routes_count_2 = len(app.routes)

        # Should not add duplicate routes
        assert routes_count_1 == routes_count_2


class TestHealthCheck:
    """Test health check functionality."""

    def test_service_health_check(self):
        """Test service health check."""
        service = get_service()

        health = service.health_check()

        assert health["status"] == "healthy"
        assert health["service"] == "PurchasedGoodsServicesService"

    def test_health_check_includes_engine_status(self):
        """Test health check includes all engine statuses."""
        service = get_service()

        health = service.health_check()

        assert "engines" in health
        assert "spend_based" in health["engines"]
        assert "average_data" in health["engines"]
        assert "supplier_specific" in health["engines"]
        assert "hybrid_aggregator" in health["engines"]
        assert "compliance_checker" in health["engines"]
        assert "pipeline" in health["engines"]


class TestOperationStatistics:
    """Test operation statistics tracking."""

    def test_operation_counter_increments(self):
        """Test operation counters increment correctly."""
        service = get_service()

        initial_stats = service.get_statistics()
        initial_count = initial_stats.get("spend_based_calculations", 0)

        # Perform a calculation
        input_data = SpendBasedInput(
            spend_amount=Decimal("100000"),
            category="raw_materials",
            reporting_year=2024,
        )
        service.calculate_spend_based(input_data)

        updated_stats = service.get_statistics()
        updated_count = updated_stats.get("spend_based_calculations", 0)

        assert updated_count == initial_count + 1

    def test_multiple_operation_counters(self):
        """Test multiple operation type counters."""
        service = get_service()

        # Perform different operations
        spend_input = SpendBasedInput(
            spend_amount=Decimal("100000"),
            category="raw_materials",
            reporting_year=2024,
        )
        service.calculate_spend_based(spend_input)

        from greenlang.purchased_goods_services.supplier_specific_calculator import ProductEmissions
        product = ProductEmissions(
            product_id="P001",
            quantity=Decimal("1000"),
            unit="kg",
            emission_factor=Decimal("2.5"),
        )
        service.calculate_supplier_specific(product)

        stats = service.get_statistics()

        assert stats.get("spend_based_calculations", 0) > 0
        assert stats.get("supplier_specific_calculations", 0) > 0

    def test_statistics_reset(self):
        """Test statistics can be reset."""
        service = get_service()

        # Perform operations
        input_data = SpendBasedInput(
            spend_amount=Decimal("100000"),
            category="raw_materials",
            reporting_year=2024,
        )
        service.calculate_spend_based(input_data)

        # Reset
        service.reset_statistics()

        stats = service.get_statistics()
        assert stats.get("spend_based_calculations", 0) == 0


class TestErrorHandling:
    """Test error handling in service."""

    def test_invalid_input_raises_error(self):
        """Test invalid input raises appropriate error."""
        service = get_service()

        with pytest.raises(ValueError):
            service.calculate_spend_based(None)

    def test_missing_required_field_raises_error(self):
        """Test missing required field raises error."""
        service = get_service()

        # Invalid input - missing required fields
        with pytest.raises((ValueError, TypeError)):
            service.calculate_hybrid([])

    def test_engine_failure_handled_gracefully(self):
        """Test engine failure is handled gracefully."""
        service = get_service()

        with patch.object(service, '_spend_based_engine', side_effect=Exception("Engine error")):
            with pytest.raises(Exception):
                input_data = SpendBasedInput(
                    spend_amount=Decimal("100000"),
                    category="raw_materials",
                    reporting_year=2024,
                )
                service.calculate_spend_based(input_data)


class TestCaching:
    """Test caching behavior."""

    def test_eeio_factor_caching(self):
        """Test EEIO factor lookup is cached."""
        service = get_service()

        # First lookup
        factor1 = service.lookup_eeio_factor(
            category="raw_materials",
            taxonomy="naics",
            year=2024,
        )

        # Second lookup (should be cached)
        factor2 = service.lookup_eeio_factor(
            category="raw_materials",
            taxonomy="naics",
            year=2024,
        )

        assert factor1 == factor2

    def test_avgdata_factor_caching(self):
        """Test average-data factor lookup is cached."""
        service = get_service()

        # First lookup
        factor1 = service.lookup_avgdata_factor(
            product_category="steel",
            region="US",
            year=2024,
        )

        # Second lookup (should be cached)
        factor2 = service.lookup_avgdata_factor(
            product_category="steel",
            region="US",
            year=2024,
        )

        assert factor1 == factor2

    def test_cache_invalidation(self):
        """Test cache can be invalidated."""
        service = get_service()

        # Lookup and cache
        factor1 = service.lookup_eeio_factor(
            category="raw_materials",
            taxonomy="naics",
            year=2024,
        )

        # Clear cache
        service.clear_cache()

        # Lookup again (should re-fetch)
        factor2 = service.lookup_eeio_factor(
            category="raw_materials",
            taxonomy="naics",
            year=2024,
        )

        # Values should still be the same, but fetched fresh
        assert factor1 == factor2


class TestDatabaseIntegration:
    """Test database integration."""

    @patch('greenlang.purchased_goods_services.setup.get_db_connection')
    def test_database_connection_acquired(self, mock_get_db):
        """Test database connection is acquired when needed."""
        service = get_service()

        mock_conn = Mock()
        mock_get_db.return_value = mock_conn

        # Operation that requires DB
        service.lookup_eeio_factor(
            category="raw_materials",
            taxonomy="naics",
            year=2024,
        )

        # Verify DB connection was requested
        assert mock_get_db.called

    @patch('greenlang.purchased_goods_services.setup.get_db_connection')
    def test_database_error_handling(self, mock_get_db):
        """Test database errors are handled properly."""
        service = get_service()

        mock_get_db.side_effect = Exception("Database connection error")

        with pytest.raises(Exception):
            service.lookup_eeio_factor(
                category="raw_materials",
                taxonomy="naics",
                year=2024,
            )


class TestConcurrency:
    """Test concurrent operations."""

    def test_concurrent_calculations(self):
        """Test service handles concurrent calculations."""
        import threading
        service = get_service()

        results = []

        def calculate():
            input_data = SpendBasedInput(
                spend_amount=Decimal("100000"),
                category="raw_materials",
                reporting_year=2024,
            )
            result = service.calculate_spend_based(input_data)
            results.append(result)

        threads = [threading.Thread(target=calculate) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(r["total_emissions"] > 0 for r in results)

    def test_thread_safe_statistics(self):
        """Test statistics are thread-safe."""
        import threading
        service = get_service()

        service.reset_statistics()

        def increment_counter():
            input_data = SpendBasedInput(
                spend_amount=Decimal("100000"),
                category="raw_materials",
                reporting_year=2024,
            )
            service.calculate_spend_based(input_data)

        threads = [threading.Thread(target=increment_counter) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = service.get_statistics()
        assert stats.get("spend_based_calculations", 0) == 100
