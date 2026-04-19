"""
GL-011 FUELCRAFT - Main Optimizer (FuelOptimizationAgent) Tests

Unit tests for FuelOptimizationAgent including full pipeline
execution, input/output validation, and component integration.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

from greenlang.agents.process_heat.gl_011_fuel_optimization.optimizer import (
    FuelOptimizationAgent,
    DEFAULT_FUEL_PROPERTIES,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.config import (
    FuelOptimizationConfig,
    FuelType,
    OptimizationMode,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.schemas import (
    FuelOptimizationInput,
    FuelOptimizationOutput,
    OptimizationResult,
    FuelPrice,
)


class TestFuelOptimizationAgent:
    """Tests for FuelOptimizationAgent class."""

    def test_agent_initialization(self, fuel_optimization_config):
        """Test agent initialization."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        assert agent.fuel_config == fuel_optimization_config
        assert agent.hv_calculator is not None
        assert agent.pricing_service is not None
        assert agent.blending_optimizer is not None
        assert agent.switching_controller is not None
        assert agent.inventory_manager is not None
        assert agent.cost_optimizer is not None

    def test_agent_has_default_fuel_properties(self, fuel_optimization_config):
        """Test agent has default fuel properties."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        assert "natural_gas" in agent._fuel_properties
        assert "no2_fuel_oil" in agent._fuel_properties
        assert "lpg_propane" in agent._fuel_properties


class TestInputValidation:
    """Tests for input validation."""

    def test_valid_input_passes(self, fuel_optimization_config, fuel_optimization_input):
        """Test valid input passes validation."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.validate_input(fuel_optimization_input)

        assert result is True

    def test_missing_facility_id_fails(self, fuel_optimization_config):
        """Test missing facility_id fails validation."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        input_data = FuelOptimizationInput(
            facility_id="",  # Empty
            current_fuel="natural_gas",
            current_fuel_flow_rate=50000.0,
            current_heat_input_mmbtu_hr=50.0,
        )

        result = agent.validate_input(input_data)

        assert result is False

    def test_missing_current_fuel_fails(self, fuel_optimization_config):
        """Test missing current_fuel fails validation."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        input_data = FuelOptimizationInput(
            facility_id="PLANT-001",
            current_fuel="",  # Empty
            current_fuel_flow_rate=50000.0,
            current_heat_input_mmbtu_hr=50.0,
        )

        result = agent.validate_input(input_data)

        assert result is False

    def test_invalid_heat_input_fails(self, fuel_optimization_config):
        """Test invalid heat input fails validation."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        input_data = FuelOptimizationInput(
            facility_id="PLANT-001",
            current_fuel="natural_gas",
            current_fuel_flow_rate=50000.0,
            current_heat_input_mmbtu_hr=0.0,  # Zero/invalid
        )

        result = agent.validate_input(input_data)

        assert result is False


class TestOutputValidation:
    """Tests for output validation."""

    def test_valid_output_passes(self, fuel_optimization_config):
        """Test valid output passes validation."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        output = FuelOptimizationOutput(
            facility_id="PLANT-001",
            request_id="test-123",
            optimization_result=OptimizationResult(
                recommended_fuel_cost_usd_hr=175.0,
                current_fuel_cost_usd_hr=200.0,
            ),
            provenance_hash="abc123",
        )

        result = agent.validate_output(output)

        assert result is True

    def test_missing_optimization_result_fails(self, fuel_optimization_config):
        """Test missing optimization result fails validation."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        output = FuelOptimizationOutput(
            facility_id="PLANT-001",
            request_id="test-123",
            optimization_result=None,  # Missing
            provenance_hash="abc123",
        )

        result = agent.validate_output(output)

        assert result is False

    def test_missing_provenance_hash_fails(self, fuel_optimization_config):
        """Test missing provenance hash fails validation."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        output = FuelOptimizationOutput(
            facility_id="PLANT-001",
            request_id="test-123",
            optimization_result=OptimizationResult(
                recommended_fuel_cost_usd_hr=175.0,
                current_fuel_cost_usd_hr=200.0,
            ),
            provenance_hash="",  # Empty
        )

        result = agent.validate_output(output)

        assert result is False


class TestProcessMethod:
    """Tests for main process method."""

    def test_process_returns_output(self, fuel_optimization_config, fuel_optimization_input):
        """Test process returns FuelOptimizationOutput."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert isinstance(result, FuelOptimizationOutput)
        assert result.status == "success"

    def test_process_includes_optimization_result(self, fuel_optimization_config, fuel_optimization_input):
        """Test process includes optimization result."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert result.optimization_result is not None
        assert result.optimization_result.current_fuel_cost_usd_hr >= 0

    def test_process_includes_provenance_hash(self, fuel_optimization_config, fuel_optimization_input):
        """Test process includes provenance hash."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) > 0

    def test_process_includes_input_hash(self, fuel_optimization_config, fuel_optimization_input):
        """Test process includes input hash."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert result.input_hash is not None

    def test_process_includes_processing_time(self, fuel_optimization_config, fuel_optimization_input):
        """Test process includes processing time."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert result.processing_time_ms >= 0

    def test_process_includes_kpis(self, fuel_optimization_config, fuel_optimization_input):
        """Test process includes KPIs."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert result.kpis is not None
        assert "fuel_cost_usd_hr" in result.kpis
        assert "heat_input_mmbtu_hr" in result.kpis


class TestFuelPriceHandling:
    """Tests for fuel price handling."""

    def test_uses_provided_prices(self, fuel_optimization_config, all_fuel_prices):
        """Test agent uses provided fuel prices."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        input_data = FuelOptimizationInput(
            facility_id="PLANT-001",
            current_fuel="natural_gas",
            current_fuel_flow_rate=50000.0,
            current_heat_input_mmbtu_hr=50.0,
            fuel_prices=all_fuel_prices,
        )

        result = agent.process(input_data)

        # Should use provided prices
        assert "natural_gas" in result.fuel_prices_used

    def test_fetches_prices_when_not_provided(self, fuel_optimization_config):
        """Test agent fetches prices when not provided."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        input_data = FuelOptimizationInput(
            facility_id="PLANT-001",
            current_fuel="natural_gas",
            current_fuel_flow_rate=50000.0,
            current_heat_input_mmbtu_hr=50.0,
            fuel_prices=None,  # Not provided
        )

        result = agent.process(input_data)

        # Should still have prices
        assert len(result.fuel_prices_used) > 0


class TestBlendingIntegration:
    """Tests for blending optimizer integration."""

    def test_blending_evaluated_when_enabled(self, fuel_optimization_config, fuel_optimization_input):
        """Test blending is evaluated when enabled."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        # Blending may or may not have a recommendation
        # but the evaluation should have run
        assert result.optimization_result is not None

    def test_no_blending_when_disabled(self, fuel_optimization_config, fuel_optimization_input):
        """Test no blending when disabled."""
        fuel_optimization_config.blending.enabled = False
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert result.optimization_result.blend_recommendation is None


class TestSwitchingIntegration:
    """Tests for switching controller integration."""

    def test_switching_evaluated(self, fuel_optimization_config, fuel_optimization_input):
        """Test switching is evaluated."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        # Switching evaluation should have run
        assert result.optimization_result is not None

    def test_no_switching_when_disabled(self, fuel_optimization_config, fuel_optimization_input):
        """Test no switching when disabled."""
        from greenlang.agents.process_heat.gl_011_fuel_optimization.config import SwitchingMode

        fuel_optimization_config.switching.mode = SwitchingMode.DISABLED
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert result.optimization_result.switching_recommendation is None


class TestCostOptimizationIntegration:
    """Tests for cost optimizer integration."""

    def test_cost_optimization_runs(self, fuel_optimization_config, fuel_optimization_input):
        """Test cost optimization runs."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        # Cost analysis should be present
        assert result.optimization_result.cost_analysis is not None or \
               result.optimization_result.current_fuel_cost_usd_hr > 0


class TestInventoryIntegration:
    """Tests for inventory manager integration."""

    def test_inventory_alerts_included(self, fuel_optimization_config, fuel_optimization_input):
        """Test inventory alerts are included."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        # inventory_alerts should be a list (may be empty)
        assert isinstance(result.inventory_alerts, list)


class TestDeliveryRecommendations:
    """Tests for delivery recommendations."""

    def test_delivery_recommendations_included(self, fuel_optimization_config, fuel_optimization_input):
        """Test delivery recommendations are included."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        # delivery_recommendations should be a list (may be empty)
        assert isinstance(result.delivery_recommendations, list)


class TestKPICalculation:
    """Tests for KPI calculation."""

    def test_fuel_cost_kpi(self, fuel_optimization_config, fuel_optimization_input):
        """Test fuel cost KPI is calculated."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert "fuel_cost_usd_hr" in result.kpis
        assert result.kpis["fuel_cost_usd_hr"] >= 0

    def test_savings_kpi(self, fuel_optimization_config, fuel_optimization_input):
        """Test potential savings KPI."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert "potential_savings_usd_hr" in result.kpis

    def test_co2_kpi(self, fuel_optimization_config, fuel_optimization_input):
        """Test CO2 emissions KPI."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert "co2_emissions_kg_hr" in result.kpis
        assert result.kpis["co2_emissions_kg_hr"] >= 0


class TestEmissionFactorLookup:
    """Tests for emission factor lookup."""

    def test_get_emission_factor_natural_gas(self, fuel_optimization_config):
        """Test emission factor lookup for natural gas."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        factor = agent._get_emission_factor("natural_gas")

        assert factor == pytest.approx(53.06, rel=0.01)

    def test_get_emission_factor_unknown_defaults(self, fuel_optimization_config):
        """Test unknown fuel defaults to natural gas factor."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        factor = agent._get_emission_factor("unknown_fuel")

        assert factor == pytest.approx(53.06, rel=0.01)


class TestOptimizationResult:
    """Tests for optimization result creation."""

    def test_result_includes_current_cost(self, fuel_optimization_config, fuel_optimization_input):
        """Test result includes current fuel cost."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert result.optimization_result.current_fuel_cost_usd_hr > 0

    def test_result_includes_recommended_cost(self, fuel_optimization_config, fuel_optimization_input):
        """Test result includes recommended fuel cost."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert result.optimization_result.recommended_fuel_cost_usd_hr >= 0

    def test_result_includes_co2_fields(self, fuel_optimization_config, fuel_optimization_input):
        """Test result includes CO2 fields."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert result.optimization_result.current_co2_kg_hr >= 0
        assert result.optimization_result.recommended_co2_kg_hr >= 0


class TestMetadata:
    """Tests for output metadata."""

    def test_metadata_includes_version(self, fuel_optimization_config, fuel_optimization_input):
        """Test metadata includes agent version."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert "agent_version" in result.metadata
        assert result.metadata["agent_version"] == fuel_optimization_config.agent_version

    def test_metadata_includes_optimization_mode(self, fuel_optimization_config, fuel_optimization_input):
        """Test metadata includes optimization mode."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert "optimization_mode" in result.metadata


class TestProvenanceHash:
    """Tests for provenance hash calculation."""

    def test_provenance_hash_is_sha256(self, fuel_optimization_config, fuel_optimization_input):
        """Test provenance hash is SHA-256 format."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result = agent.process(fuel_optimization_input)

        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_provenance_hash_deterministic(self, fuel_optimization_config, fuel_optimization_input):
        """Test same input produces same hash."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        result1 = agent.process(fuel_optimization_input)
        result2 = agent.process(fuel_optimization_input)

        assert result1.provenance_hash == result2.provenance_hash


class TestErrorHandling:
    """Tests for error handling."""

    def test_validation_error_raised(self, fuel_optimization_config):
        """Test ValidationError raised for invalid input."""
        from greenlang.agents.process_heat.shared.base_agent import ValidationError

        agent = FuelOptimizationAgent(fuel_optimization_config)

        input_data = FuelOptimizationInput(
            facility_id="",  # Invalid
            current_fuel="natural_gas",
            current_fuel_flow_rate=50000.0,
            current_heat_input_mmbtu_hr=50.0,
        )

        with pytest.raises(ValidationError):
            agent.process(input_data)


class TestDefaultFuelProperties:
    """Tests for default fuel properties."""

    def test_natural_gas_properties(self):
        """Test default natural gas properties."""
        props = DEFAULT_FUEL_PROPERTIES["natural_gas"]

        assert props.fuel_type == "natural_gas"
        assert props.hhv_btu_scf == pytest.approx(1020.0, rel=0.01)
        assert props.co2_kg_mmbtu == pytest.approx(53.06, rel=0.01)

    def test_fuel_oil_properties(self):
        """Test default fuel oil properties."""
        props = DEFAULT_FUEL_PROPERTIES["no2_fuel_oil"]

        assert props.fuel_type == "no2_fuel_oil"
        assert props.hhv_btu_lb == pytest.approx(19580, rel=0.01)

    def test_hydrogen_properties(self):
        """Test default hydrogen properties."""
        props = DEFAULT_FUEL_PROPERTIES["hydrogen"]

        assert props.fuel_type == "hydrogen"
        assert props.co2_kg_mmbtu == 0.0


class TestPriceConversion:
    """Tests for price conversion utilities."""

    def test_price_quote_to_schema(self, fuel_optimization_config):
        """Test converting PriceQuote to FuelPrice schema."""
        from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_pricing import PriceQuote

        agent = FuelOptimizationAgent(fuel_optimization_config)

        quote = PriceQuote(
            fuel_type="natural_gas",
            commodity_price=3.00,
            basis_differential=0.20,
            transport_cost=0.15,
            taxes=0.15,
            carbon_cost=0.0,
            total_price=3.50,
            source="henry_hub",
            valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
            confidence=0.95,
        )

        price = agent._price_quote_to_schema(quote)

        assert isinstance(price, FuelPrice)
        assert price.fuel_type == "natural_gas"
        assert price.price == 3.50
        assert price.confidence == 0.95

    def test_schema_to_price_quote(self, fuel_optimization_config, natural_gas_price):
        """Test converting FuelPrice schema to PriceQuote."""
        from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_pricing import PriceQuote

        agent = FuelOptimizationAgent(fuel_optimization_config)

        quote = agent._schema_to_price_quote(natural_gas_price)

        assert isinstance(quote, PriceQuote)
        assert quote.fuel_type == "natural_gas"
        assert quote.total_price == natural_gas_price.price


class TestIntelligenceMixin:
    """Tests for IntelligenceMixin integration."""

    def test_get_intelligence_level(self, fuel_optimization_config):
        """Test intelligence level is returned."""
        from greenlang.agents.intelligence_interface import IntelligenceLevel

        agent = FuelOptimizationAgent(fuel_optimization_config)

        level = agent.get_intelligence_level()

        assert level == IntelligenceLevel.STANDARD

    def test_get_intelligence_capabilities(self, fuel_optimization_config):
        """Test intelligence capabilities are returned."""
        agent = FuelOptimizationAgent(fuel_optimization_config)

        capabilities = agent.get_intelligence_capabilities()

        assert capabilities.can_explain is True
        assert capabilities.can_recommend is True
        assert capabilities.can_validate is True
