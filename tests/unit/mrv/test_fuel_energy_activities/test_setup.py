"""
Unit tests for FuelEnergyActivitiesService

Tests service layer, singleton pattern, and setup functions.
Validates health checks, statistics, and router configuration.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch
from fastapi import FastAPI

from greenlang.agents.mrv.fuel_energy_activities.setup import (
    FuelEnergyActivitiesService,
    configure_fuel_energy_activities,
    get_service,
    get_router,
)
from greenlang.agents.mrv.fuel_energy_activities.models import FuelType, ActivityType
from greenlang_core import AgentConfig
from greenlang_core.exceptions import ValidationError


# Fixtures
@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return AgentConfig(
        name="fuel_energy_activities",
        version="1.0.0",
        environment="test"
    )


@pytest.fixture
def service(agent_config):
    """Create FuelEnergyActivitiesService instance."""
    return FuelEnergyActivitiesService(agent_config)


@pytest.fixture
def app():
    """Create FastAPI app for testing."""
    return FastAPI()


# Test Class
class TestFuelEnergyActivitiesService:
    """Test suite for FuelEnergyActivitiesService."""

    def test_singleton(self, agent_config):
        """Test service follows singleton pattern."""
        service1 = FuelEnergyActivitiesService(agent_config)
        service2 = FuelEnergyActivitiesService(agent_config)

        # Should return same instance
        assert service1 is service2

    def test_calculate_activity_3a(self, service):
        """Test calculating activity 3a (upstream of purchased fuels)."""
        input_data = {
            "fuel_type": FuelType.NATURAL_GAS,
            "quantity": Decimal("1000"),
            "country": "US",
            "reporting_period": "2025-Q1"
        }

        result = service.calculate_activity_3a(input_data)

        assert result is not None
        assert result.total_emissions_kgco2e > Decimal("0")
        assert result.activity_type == ActivityType.ACTIVITY_3A

    def test_calculate_activity_3b(self, service):
        """Test calculating activity 3b (upstream of purchased electricity)."""
        input_data = {
            "electricity_kwh": Decimal("100000"),
            "country": "US",
            "reporting_period": "2025-Q1"
        }

        result = service.calculate_activity_3b(input_data)

        assert result is not None
        assert result.total_emissions_kgco2e > Decimal("0")
        assert result.activity_type == ActivityType.ACTIVITY_3B

    def test_calculate_activity_3c(self, service):
        """Test calculating activity 3c (T&D losses)."""
        input_data = {
            "electricity_consumption_kwh": Decimal("100000"),
            "country": "US",
            "reporting_period": "2025-Q1"
        }

        result = service.calculate_activity_3c(input_data)

        assert result is not None
        assert result.total_emissions_kgco2e > Decimal("0")
        assert result.activity_type == ActivityType.ACTIVITY_3C

    def test_calculate_all(self, service):
        """Test calculating all activities via pipeline."""
        input_data = {
            "fuel_consumptions": [
                {
                    "fuel_type": FuelType.NATURAL_GAS,
                    "quantity": Decimal("1000"),
                    "country": "US"
                }
            ],
            "electricity_consumptions": [
                {
                    "electricity_kwh": Decimal("50000"),
                    "country": "US"
                }
            ],
            "td_loss_calculations": [
                {
                    "electricity_consumption_kwh": Decimal("50000"),
                    "country": "US"
                }
            ],
            "reporting_period": "2025-Q1"
        }

        result = service.calculate_all(input_data)

        assert result is not None
        assert result.total_emissions_kgco2e > Decimal("0")
        assert result.activity_3a_emissions_kgco2e > Decimal("0")
        assert result.activity_3b_emissions_kgco2e > Decimal("0")
        assert result.activity_3c_emissions_kgco2e > Decimal("0")

    def test_get_wtt_factor(self, service):
        """Test getting WTT emission factor."""
        wtt_factor = service.get_wtt_factor(FuelType.NATURAL_GAS, "US")

        assert wtt_factor > Decimal("0")
        assert wtt_factor < Decimal("20")  # Reasonable range for WTT

    def test_get_upstream_ef(self, service):
        """Test getting upstream emission factor."""
        upstream_ef = service.get_upstream_ef("US")

        assert upstream_ef > Decimal("0")
        assert upstream_ef < Decimal("1")  # kg CO2e per kWh

    def test_get_td_loss_factor(self, service):
        """Test getting T&D loss factor."""
        td_loss_factor = service.get_td_loss_factor("US")

        assert td_loss_factor > Decimal("0")
        assert td_loss_factor < Decimal("0.30")  # <30%

    def test_get_available_fuels(self, service):
        """Test getting list of available fuel types."""
        fuels = service.get_available_fuels()

        assert isinstance(fuels, list)
        assert len(fuels) > 0
        assert FuelType.NATURAL_GAS in fuels
        assert FuelType.DIESEL in fuels
        assert FuelType.GASOLINE in fuels

    def test_convert_fuel_units(self, service):
        """Test converting fuel units."""
        # Convert 1000 liters of diesel to gallons
        result = service.convert_fuel_units(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("1000"),
            from_unit="LITERS",
            to_unit="GALLONS"
        )

        # 1000 liters ≈ 264.17 gallons
        assert result == pytest.approx(Decimal("264.17"), rel=Decimal("0.01"))

    def test_check_compliance(self, service):
        """Test compliance checking."""
        input_data = {
            "framework": "GHG_PROTOCOL",
            "activity_3a_emissions_kgco2e": Decimal("50000"),
            "activity_3b_emissions_kgco2e": Decimal("30000"),
            "activity_3c_emissions_kgco2e": Decimal("10000"),
            "reporting_period": "2025-Q1"
        }

        result = service.check_compliance(input_data)

        assert result is not None
        assert result.framework is not None
        assert result.compliance_status is not None

    def test_assess_dqi(self, service):
        """Test data quality index assessment."""
        input_data = {
            "fuel_type": FuelType.NATURAL_GAS,
            "quantity": Decimal("1000"),
            "country": "US",
            "verification_level": "THIRD_PARTY_VERIFIED"
        }

        dqi_score = service.assess_dqi(input_data)

        assert dqi_score is not None
        assert Decimal("0") <= dqi_score <= Decimal("5")

        # Third-party verified should have high DQI
        assert dqi_score >= Decimal("3.5")

    def test_quantify_uncertainty(self, service):
        """Test uncertainty quantification."""
        input_data = {
            "fuel_type": FuelType.NATURAL_GAS,
            "quantity": Decimal("1000"),
            "country": "US"
        }

        uncertainty_pct = service.quantify_uncertainty(input_data)

        assert uncertainty_pct is not None
        assert uncertainty_pct > Decimal("0")
        assert uncertainty_pct < Decimal("100")

    def test_health_check(self, service):
        """Test health check."""
        health = service.health_check()

        assert health is not None
        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "wtt_calculator" in health
        assert "upstream_calculator" in health
        assert "td_loss_calculator" in health

    def test_get_statistics(self, service):
        """Test getting service statistics."""
        # Perform some calculations
        service.calculate_activity_3a({
            "fuel_type": FuelType.NATURAL_GAS,
            "quantity": Decimal("1000"),
            "country": "US",
            "reporting_period": "2025-Q1"
        })

        stats = service.get_statistics()

        assert stats is not None
        assert "calculations_performed" in stats
        assert stats["calculations_performed"] > 0

    def test_get_service_returns_singleton(self, agent_config):
        """Test get_service() returns singleton instance."""
        service1 = get_service(agent_config)
        service2 = get_service(agent_config)

        assert service1 is service2

    def test_get_router_returns_api_router(self):
        """Test get_router() returns FastAPI router."""
        router = get_router()

        assert router is not None
        # Should have routes
        assert len(router.routes) > 0

    def test_configure_fuel_energy_activities(self, app, agent_config):
        """Test configuring fuel & energy activities on FastAPI app."""
        configure_fuel_energy_activities(app, agent_config)

        # Should add router to app
        # Check if routes were added
        route_paths = [route.path for route in app.routes]

        # Should have fuel & energy routes
        assert any("/fuel-energy" in path for path in route_paths)

    def test_batch_calculation(self, service):
        """Test batch calculation."""
        inputs = [
            {
                "fuel_type": FuelType.NATURAL_GAS,
                "quantity": Decimal("1000"),
                "country": "US",
                "reporting_period": "2025-Q1"
            },
            {
                "fuel_type": FuelType.DIESEL,
                "quantity": Decimal("500"),
                "country": "GB",
                "reporting_period": "2025-Q1"
            },
        ]

        results = service.calculate_activity_3a_batch(inputs)

        assert len(results) == 2
        assert all(r.activity_type == ActivityType.ACTIVITY_3A for r in results)

    def test_aggregate_by_fuel_type(self, service):
        """Test aggregating by fuel type."""
        inputs = [
            {
                "fuel_type": FuelType.NATURAL_GAS,
                "quantity": Decimal("1000"),
                "country": "US",
                "reporting_period": "2025-Q1"
            },
            {
                "fuel_type": FuelType.NATURAL_GAS,
                "quantity": Decimal("500"),
                "country": "GB",
                "reporting_period": "2025-Q1"
            },
            {
                "fuel_type": FuelType.DIESEL,
                "quantity": Decimal("300"),
                "country": "US",
                "reporting_period": "2025-Q1"
            },
        ]

        results = service.calculate_activity_3a_batch(inputs)
        aggregated = service.aggregate_by_fuel_type(results)

        assert FuelType.NATURAL_GAS in aggregated
        assert FuelType.DIESEL in aggregated

        # Natural gas should have combined quantity
        assert aggregated[FuelType.NATURAL_GAS]["total_quantity"] == Decimal("1500")

    def test_aggregate_by_country(self, service):
        """Test aggregating by country."""
        inputs = [
            {
                "fuel_type": FuelType.NATURAL_GAS,
                "quantity": Decimal("1000"),
                "country": "US",
                "reporting_period": "2025-Q1"
            },
            {
                "fuel_type": FuelType.DIESEL,
                "quantity": Decimal("500"),
                "country": "US",
                "reporting_period": "2025-Q1"
            },
            {
                "fuel_type": FuelType.NATURAL_GAS,
                "quantity": Decimal("800"),
                "country": "GB",
                "reporting_period": "2025-Q1"
            },
        ]

        results = service.calculate_activity_3a_batch(inputs)
        aggregated = service.aggregate_by_country(results)

        assert "US" in aggregated
        assert "GB" in aggregated

    def test_compare_with_baseline(self, service):
        """Test comparing with baseline year."""
        current_input = {
            "fuel_consumptions": [
                {"fuel_type": FuelType.NATURAL_GAS, "quantity": Decimal("1000"), "country": "US"}
            ],
            "reporting_period": "2025-Q1"
        }

        baseline_input = {
            "fuel_consumptions": [
                {"fuel_type": FuelType.NATURAL_GAS, "quantity": Decimal("1200"), "country": "US"}
            ],
            "reporting_period": "2020-Q1"
        }

        current_result = service.calculate_all(current_input)
        baseline_result = service.calculate_all(baseline_input)

        comparison = service.compare_with_baseline(current_result, baseline_result)

        assert "current_emissions" in comparison
        assert "baseline_emissions" in comparison
        assert "reduction_pct" in comparison
        assert "reduction_kgco2e" in comparison

    def test_export_to_json(self, service):
        """Test exporting to JSON."""
        result = service.calculate_activity_3a({
            "fuel_type": FuelType.NATURAL_GAS,
            "quantity": Decimal("1000"),
            "country": "US",
            "reporting_period": "2025-Q1"
        })

        json_output = service.export_to_json(result)

        assert isinstance(json_output, str)
        import json
        parsed = json.loads(json_output)
        assert "total_emissions_kgco2e" in parsed

    def test_export_to_csv(self, service):
        """Test exporting to CSV."""
        results = [
            service.calculate_activity_3a({
                "fuel_type": FuelType.NATURAL_GAS,
                "quantity": Decimal("1000"),
                "country": "US",
                "reporting_period": "2025-Q1"
            })
        ]

        csv_output = service.export_to_csv(results)

        assert isinstance(csv_output, str)
        assert "total_emissions_kgco2e" in csv_output

    def test_error_handling_invalid_fuel_type(self, service):
        """Test error handling for invalid fuel type."""
        input_data = {
            "fuel_type": "INVALID_FUEL",
            "quantity": Decimal("1000"),
            "country": "US",
            "reporting_period": "2025-Q1"
        }

        with pytest.raises(ValidationError, match="fuel"):
            service.calculate_activity_3a(input_data)

    def test_error_handling_negative_quantity(self, service):
        """Test error handling for negative quantity."""
        input_data = {
            "fuel_type": FuelType.NATURAL_GAS,
            "quantity": Decimal("-1000"),
            "country": "US",
            "reporting_period": "2025-Q1"
        }

        with pytest.raises(ValidationError, match="quantity"):
            service.calculate_activity_3a(input_data)

    def test_cache_hit(self, service):
        """Test cache hit for repeated calculations."""
        input_data = {
            "fuel_type": FuelType.NATURAL_GAS,
            "quantity": Decimal("1000"),
            "country": "US",
            "reporting_period": "2025-Q1"
        }

        # First calculation
        result1 = service.calculate_activity_3a(input_data)

        # Second calculation (should hit cache)
        result2 = service.calculate_activity_3a(input_data)

        # Results should be identical
        assert result1.provenance_hash == result2.provenance_hash

    def test_get_emission_factor_sources(self, service):
        """Test getting emission factor sources."""
        sources = service.get_emission_factor_sources()

        assert isinstance(sources, list)
        assert len(sources) > 0

        # Should include common sources
        source_names = [s["name"] for s in sources]
        assert any("DEFRA" in name for name in source_names)

    def test_get_supported_countries(self, service):
        """Test getting supported countries."""
        countries = service.get_supported_countries()

        assert isinstance(countries, list)
        assert len(countries) >= 50  # Should support 50+ countries

        assert "US" in countries
        assert "GB" in countries
        assert "DE" in countries

    def test_reset_service(self, service):
        """Test resetting service state."""
        # Perform calculation
        service.calculate_activity_3a({
            "fuel_type": FuelType.NATURAL_GAS,
            "quantity": Decimal("1000"),
            "country": "US",
            "reporting_period": "2025-Q1"
        })

        # Reset
        service.reset()

        stats = service.get_statistics()
        assert stats["calculations_performed"] == 0

    def test_performance_single_calculation(self, service, benchmark):
        """Test single calculation performance."""
        input_data = {
            "fuel_type": FuelType.NATURAL_GAS,
            "quantity": Decimal("1000"),
            "country": "US",
            "reporting_period": "2025-Q1"
        }

        def run_calculation():
            return service.calculate_activity_3a(input_data)

        result = benchmark(run_calculation)

        assert result is not None


# Integration Tests
class TestFuelEnergyActivitiesServiceIntegration:
    """Integration tests for FuelEnergyActivitiesService."""

    @pytest.mark.integration
    def test_integration_with_database(self, service):
        """Test integration with database."""
        # This would test actual database operations
        pass

    @pytest.mark.integration
    def test_integration_with_cache(self, service):
        """Test integration with Redis cache."""
        # This would test actual cache operations
        pass


# Performance Tests
class TestFuelEnergyActivitiesServicePerformance:
    """Performance tests for FuelEnergyActivitiesService."""

    @pytest.mark.performance
    def test_throughput_target(self, service):
        """Test service meets throughput target."""
        num_records = 10000

        inputs = [
            {
                "fuel_type": FuelType.NATURAL_GAS,
                "quantity": Decimal("1000"),
                "country": "US",
                "reporting_period": "2025-Q1"
            }
            for _ in range(num_records)
        ]

        start_time = datetime.now()
        results = service.calculate_activity_3a_batch(inputs)
        end_time = datetime.now()

        duration_seconds = (end_time - start_time).total_seconds()
        throughput = num_records / duration_seconds

        assert throughput >= 1000  # Target: 1000 calculations/sec
        assert len(results) == num_records
