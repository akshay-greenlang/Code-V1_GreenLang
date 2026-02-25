"""
Unit tests for CapitalGoodsService.

Tests service layer integration, calculation methods, asset management,
health checks, and statistics reporting.
"""

import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any

from greenlang.mrv.capital_goods.setup import (
    CapitalGoodsService,
    CalculationRequest,
    CalculationResponse,
    AssetRegistration,
    EmissionFactorResponse,
)


class TestCapitalGoodsServiceSingleton:
    """Test singleton pattern."""

    def test_singleton_same_instance(self):
        """Test that multiple calls return same instance."""
        service1 = CapitalGoodsService()
        service2 = CapitalGoodsService()
        assert service1 is service2

    def test_singleton_with_reset(self):
        """Test singleton reset for testing."""
        service1 = CapitalGoodsService()
        CapitalGoodsService._instance = None
        service2 = CapitalGoodsService()
        assert service1 is not service2


class TestCalculate:
    """Test calculate() integration method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    @pytest.fixture
    def sample_request(self):
        """Create sample calculation request."""
        return CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[
                {
                    "asset_id": "A001",
                    "asset_type": "Server",
                    "purchase_value": 10000.00,
                    "purchase_date": "2024-06-01",
                }
            ],
            calculation_method="spend_based",
            frameworks=["ghg_protocol"],
        )

    def test_calculate_success(self, service, sample_request):
        """Test successful calculation."""
        response = service.calculate(sample_request)

        assert isinstance(response, CalculationResponse)
        assert response.calculation_id is not None
        assert response.total_emissions >= 0
        assert response.status in ["completed", "completed_with_warnings"]

    def test_calculate_returns_provenance_hash(self, service, sample_request):
        """Test calculation returns provenance hash."""
        response = service.calculate(sample_request)

        assert response.provenance_hash is not None
        assert len(response.provenance_hash) == 64  # SHA-256

    def test_calculate_stores_result(self, service, sample_request):
        """Test calculation result is stored."""
        response = service.calculate(sample_request)

        # Should be able to retrieve stored calculation
        stored = service.get_calculation(response.calculation_id)
        assert stored is not None
        assert stored["calculation_id"] == response.calculation_id


class TestCalculateBatch:
    """Test calculate_batch() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_calculate_batch_multiple_requests(self, service):
        """Test batch calculation with multiple requests."""
        requests = [
            CalculationRequest(
                organization_id=f"ORG{i:03d}",
                reporting_period_start=date(2024, 1, 1),
                reporting_period_end=date(2024, 12, 31),
                assets=[
                    {"asset_id": "A001", "asset_type": "Equipment", "purchase_value": 5000.00}
                ],
                calculation_method="spend_based",
                frameworks=["ghg_protocol"],
            )
            for i in range(5)
        ]

        responses = service.calculate_batch(requests)

        assert len(responses) == 5
        assert all(isinstance(r, CalculationResponse) for r in responses)

    def test_calculate_batch_empty_list(self, service):
        """Test batch calculation with empty list."""
        responses = service.calculate_batch([])
        assert responses == []


class TestCalculateSpendBased:
    """Test calculate_spend_based() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_calculate_spend_based_single_asset(self, service):
        """Test spend-based calculation for single asset."""
        request = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[
                {"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}
            ],
            calculation_method="spend_based",
            frameworks=["ghg_protocol"],
        )

        response = service.calculate_spend_based(request)

        assert response.calculation_method == "spend_based"
        assert response.total_emissions > 0

    def test_calculate_spend_based_multiple_assets(self, service):
        """Test spend-based calculation for multiple assets."""
        request = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[
                {"asset_id": f"A{i:03d}", "asset_type": "Server", "purchase_value": 5000.00}
                for i in range(10)
            ],
            calculation_method="spend_based",
            frameworks=["ghg_protocol"],
        )

        response = service.calculate_spend_based(request)

        assert response.asset_count == 10


class TestCalculateAverageData:
    """Test calculate_average_data() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_calculate_average_data_with_industry_averages(self, service):
        """Test average-data calculation using industry averages."""
        request = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[
                {
                    "asset_id": "A001",
                    "asset_type": "Server",
                    "purchase_value": 10000.00,
                    "industry": "IT Equipment",
                }
            ],
            calculation_method="average_data",
            frameworks=["ghg_protocol"],
        )

        response = service.calculate_average_data(request)

        assert response.calculation_method == "average_data"
        assert response.total_emissions > 0


class TestCalculateSupplierSpecific:
    """Test calculate_supplier_specific() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_calculate_supplier_specific_with_supplier_data(self, service):
        """Test supplier-specific calculation with supplier data."""
        request = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[
                {
                    "asset_id": "A001",
                    "asset_type": "Server",
                    "purchase_value": 10000.00,
                    "supplier_id": "SUP001",
                    "emission_factor": 1.5,
                    "quantity": 5,
                }
            ],
            calculation_method="supplier_specific",
            frameworks=["ghg_protocol"],
        )

        response = service.calculate_supplier_specific(request)

        assert response.calculation_method == "supplier_specific"
        assert response.data_quality_score > 3.0  # Higher quality


class TestListCalculations:
    """Test list_calculations() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_list_calculations_returns_all(self, service):
        """Test listing all stored calculations."""
        # Create some calculations
        request = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_method="spend_based",
            frameworks=["ghg_protocol"],
        )

        service.calculate(request)
        service.calculate(request)

        calculations = service.list_calculations()

        assert len(calculations) >= 2
        assert all("calculation_id" in calc for calc in calculations)

    def test_list_calculations_filters_by_organization(self, service):
        """Test listing calculations filtered by organization."""
        request1 = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_method="spend_based",
            frameworks=["ghg_protocol"],
        )

        request2 = CalculationRequest(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A002", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_method="spend_based",
            frameworks=["ghg_protocol"],
        )

        service.calculate(request1)
        service.calculate(request2)

        org1_calcs = service.list_calculations(organization_id="ORG001")

        assert all(calc["organization_id"] == "ORG001" for calc in org1_calcs)


class TestGetCalculation:
    """Test get_calculation() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_get_calculation_exists(self, service):
        """Test retrieving existing calculation."""
        request = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_method="spend_based",
            frameworks=["ghg_protocol"],
        )

        response = service.calculate(request)
        retrieved = service.get_calculation(response.calculation_id)

        assert retrieved is not None
        assert retrieved["calculation_id"] == response.calculation_id

    def test_get_calculation_not_found(self, service):
        """Test retrieving non-existent calculation returns None."""
        retrieved = service.get_calculation("non_existent_id")
        assert retrieved is None


class TestDeleteCalculation:
    """Test delete_calculation() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_delete_calculation_success(self, service):
        """Test successful deletion of calculation."""
        request = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_method="spend_based",
            frameworks=["ghg_protocol"],
        )

        response = service.calculate(request)
        deleted = service.delete_calculation(response.calculation_id)

        assert deleted is True

        # Should no longer exist
        retrieved = service.get_calculation(response.calculation_id)
        assert retrieved is None

    def test_delete_calculation_not_found(self, service):
        """Test deleting non-existent calculation returns False."""
        deleted = service.delete_calculation("non_existent_id")
        assert deleted is False


class TestRegisterAsset:
    """Test register_asset() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_register_asset_success(self, service):
        """Test successful asset registration."""
        asset = AssetRegistration(
            asset_id="A001",
            asset_type="Server",
            purchase_value=10000.00,
            purchase_date=date(2024, 6, 1),
            organization_id="ORG001",
        )

        registered = service.register_asset(asset)

        assert registered["asset_id"] == "A001"
        assert registered["status"] == "registered"

    def test_register_asset_duplicate_fails(self, service):
        """Test registering duplicate asset fails."""
        asset = AssetRegistration(
            asset_id="A002",
            asset_type="Server",
            purchase_value=10000.00,
            purchase_date=date(2024, 6, 1),
            organization_id="ORG001",
        )

        service.register_asset(asset)

        # Try to register again
        with pytest.raises(ValueError, match="already registered"):
            service.register_asset(asset)


class TestListAssets:
    """Test list_assets() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_list_assets_returns_all(self, service):
        """Test listing all registered assets."""
        for i in range(5):
            asset = AssetRegistration(
                asset_id=f"A{i:03d}",
                asset_type="Server",
                purchase_value=10000.00,
                purchase_date=date(2024, 6, 1),
                organization_id="ORG001",
            )
            service.register_asset(asset)

        assets = service.list_assets()

        assert len(assets) >= 5

    def test_list_assets_filters_by_organization(self, service):
        """Test listing assets filtered by organization."""
        asset1 = AssetRegistration(
            asset_id="A010",
            asset_type="Server",
            purchase_value=10000.00,
            purchase_date=date(2024, 6, 1),
            organization_id="ORG001",
        )

        asset2 = AssetRegistration(
            asset_id="A020",
            asset_type="Server",
            purchase_value=10000.00,
            purchase_date=date(2024, 6, 1),
            organization_id="ORG002",
        )

        service.register_asset(asset1)
        service.register_asset(asset2)

        org1_assets = service.list_assets(organization_id="ORG001")

        assert all(asset["organization_id"] == "ORG001" for asset in org1_assets)


class TestUpdateAsset:
    """Test update_asset() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_update_asset_success(self, service):
        """Test successful asset update."""
        asset = AssetRegistration(
            asset_id="A100",
            asset_type="Server",
            purchase_value=10000.00,
            purchase_date=date(2024, 6, 1),
            organization_id="ORG001",
        )

        service.register_asset(asset)

        # Update purchase value
        updated = service.update_asset("A100", {"purchase_value": 12000.00})

        assert updated["purchase_value"] == 12000.00

    def test_update_asset_not_found(self, service):
        """Test updating non-existent asset returns None."""
        updated = service.update_asset("non_existent_id", {"purchase_value": 5000.00})
        assert updated is None


class TestGetEmissionFactors:
    """Test get_emission_factors() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_get_emission_factors_by_asset_type(self, service):
        """Test retrieving emission factors by asset type."""
        factors = service.get_emission_factors(asset_type="Server")

        assert isinstance(factors, list)
        assert len(factors) > 0
        assert all("emission_factor" in ef for ef in factors)

    def test_get_emission_factors_by_region(self, service):
        """Test retrieving emission factors by region."""
        factors = service.get_emission_factors(region="North America")

        assert isinstance(factors, list)

    def test_get_emission_factors_all(self, service):
        """Test retrieving all emission factors."""
        factors = service.get_emission_factors()

        assert isinstance(factors, list)
        assert len(factors) > 0


class TestRegisterCustomEF:
    """Test register_custom_ef() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_register_custom_ef_success(self, service):
        """Test successful registration of custom emission factor."""
        custom_ef = {
            "asset_type": "Custom Equipment",
            "emission_factor": 2.5,
            "unit": "kgCO2e/USD",
            "source": "Internal Study",
            "region": "North America",
        }

        registered = service.register_custom_ef(custom_ef)

        assert registered["status"] == "registered"
        assert registered["emission_factor"] == 2.5

    def test_register_custom_ef_validation(self, service):
        """Test custom EF registration validates required fields."""
        invalid_ef = {
            "asset_type": "Equipment",
            # Missing emission_factor
        }

        with pytest.raises(ValueError, match="emission_factor"):
            service.register_custom_ef(invalid_ef)


class TestClassifyAssets:
    """Test classify_assets() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_classify_assets_by_type(self, service):
        """Test classifying assets by type."""
        assets = [
            {"asset_id": "A001", "description": "Dell PowerEdge Server"},
            {"asset_id": "A002", "description": "Cisco Network Switch"},
        ]

        classified = service.classify_assets(assets)

        assert len(classified) == 2
        assert all("classified_type" in asset for asset in classified)

    def test_classify_assets_ml_categorization(self, service):
        """Test ML-based asset categorization."""
        assets = [
            {"asset_id": "A003", "description": "Enterprise Storage Array"},
        ]

        classified = service.classify_assets(assets, use_ml=True)

        assert classified[0]["classified_type"] is not None
        assert "confidence_score" in classified[0]


class TestCheckCompliance:
    """Test check_compliance() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_check_compliance_single_calculation(self, service):
        """Test compliance check for single calculation."""
        request = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_method="spend_based",
            frameworks=["ghg_protocol"],
        )

        response = service.calculate(request)
        compliance = service.check_compliance(response.calculation_id)

        assert "overall_status" in compliance
        assert "framework_results" in compliance

    def test_check_compliance_multiple_frameworks(self, service):
        """Test compliance check across multiple frameworks."""
        request = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_method="spend_based",
            frameworks=["ghg_protocol", "iso_14064", "csrd"],
        )

        response = service.calculate(request)
        compliance = service.check_compliance(response.calculation_id)

        assert len(compliance["framework_results"]) == 3


class TestRunUncertainty:
    """Test run_uncertainty() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_run_uncertainty_quantification(self, service):
        """Test uncertainty quantification for calculation."""
        request = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_method="spend_based",
            frameworks=["ghg_protocol"],
        )

        response = service.calculate(request)
        uncertainty = service.run_uncertainty(response.calculation_id)

        assert "relative_uncertainty" in uncertainty
        assert "lower_bound" in uncertainty
        assert "upper_bound" in uncertainty
        assert uncertainty["relative_uncertainty"] >= 0


class TestHealthCheck:
    """Test health_check() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_health_check_returns_status(self, service):
        """Test health check returns service status."""
        health = service.health_check()

        assert "status" in health
        assert "engines" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_check_engine_status(self, service):
        """Test health check includes engine statuses."""
        health = service.health_check()

        assert "spend_based_engine" in health["engines"]
        assert "average_data_engine" in health["engines"]
        assert "supplier_specific_engine" in health["engines"]
        assert "hybrid_aggregator_engine" in health["engines"]
        assert "compliance_checker_engine" in health["engines"]
        assert "pipeline_engine" in health["engines"]

    def test_health_check_database_connectivity(self, service):
        """Test health check includes database connectivity."""
        health = service.health_check()

        assert "database" in health
        assert health["database"]["connected"] in [True, False]


class TestGetStats:
    """Test get_stats() method."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        CapitalGoodsService._instance = None
        return CapitalGoodsService()

    def test_get_stats_returns_metrics(self, service):
        """Test get_stats returns service metrics."""
        # Create some calculations
        request = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": f"A{i:03d}", "asset_type": "Server", "purchase_value": 10000.00} for i in range(5)],
            calculation_method="spend_based",
            frameworks=["ghg_protocol"],
        )

        service.calculate(request)

        stats = service.get_stats()

        assert "total_calculations" in stats
        assert "total_assets" in stats
        assert "total_emissions" in stats
        assert "calculations_by_method" in stats

    def test_get_stats_calculations_by_method(self, service):
        """Test stats includes breakdown by calculation method."""
        request1 = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_method="spend_based",
            frameworks=["ghg_protocol"],
        )

        request2 = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A002", "asset_type": "Server", "purchase_value": 10000.00, "supplier_id": "SUP001"}],
            calculation_method="supplier_specific",
            frameworks=["ghg_protocol"],
        )

        service.calculate(request1)
        service.calculate(request2)

        stats = service.get_stats()

        assert "spend_based" in stats["calculations_by_method"]
        assert "supplier_specific" in stats["calculations_by_method"]

    def test_get_stats_filtered_by_organization(self, service):
        """Test stats can be filtered by organization."""
        request1 = CalculationRequest(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A001", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_method="spend_based",
            frameworks=["ghg_protocol"],
        )

        request2 = CalculationRequest(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            assets=[{"asset_id": "A002", "asset_type": "Server", "purchase_value": 10000.00}],
            calculation_method="spend_based",
            frameworks=["ghg_protocol"],
        )

        service.calculate(request1)
        service.calculate(request2)

        org1_stats = service.get_stats(organization_id="ORG001")

        # Should only count ORG001 calculations
        assert org1_stats["total_calculations"] >= 1
